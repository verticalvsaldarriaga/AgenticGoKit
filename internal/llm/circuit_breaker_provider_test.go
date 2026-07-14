package llm

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

// fakeBreaker is a minimal CircuitBreaker test double that counts calls and
// can be forced open.
type fakeBreaker struct {
	open  bool
	calls int
}

func (b *fakeBreaker) Call(fn func() error) error {
	b.calls++
	if b.open {
		return errors.New("circuit breaker is open")
	}
	return fn()
}

func noBackoff(int) time.Duration { return 0 }

func TestCircuitBreakerProvider_CallSucceedsWithoutRetry(t *testing.T) {
	inner := NewMockModelProvider()
	inner.SetCallExpectation(Response{Content: "ok"}, nil)

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 3, BackoffFunc: noBackoff},
	})

	resp, err := p.Call(context.Background(), Prompt{User: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ok" {
		t.Fatalf("got %q, want %q", resp.Content, "ok")
	}
}

func TestCircuitBreakerProvider_CallRetriesTransientThenSucceeds(t *testing.T) {
	inner := NewMockModelProvider()
	attempts := 0
	inner.SetCallFunc(func(ctx context.Context, prompt Prompt) (Response, error) {
		attempts++
		if attempts < 3 {
			return Response{}, context.DeadlineExceeded
		}
		return Response{Content: "recovered"}, nil
	})

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 3, BackoffFunc: noBackoff},
	})

	resp, err := p.Call(context.Background(), Prompt{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "recovered" {
		t.Fatalf("got %q, want %q", resp.Content, "recovered")
	}
	if attempts != 3 {
		t.Fatalf("got %d attempts, want 3", attempts)
	}
}

func TestCircuitBreakerProvider_CallDoesNotRetryNonRetryableError(t *testing.T) {
	inner := NewMockModelProvider()
	attempts := 0
	wantErr := errors.New("invalid api key")
	inner.SetCallFunc(func(ctx context.Context, prompt Prompt) (Response, error) {
		attempts++
		return Response{}, wantErr
	})

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 3, BackoffFunc: noBackoff},
	})

	_, err := p.Call(context.Background(), Prompt{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("got err %v, want %v", err, wantErr)
	}
	if attempts != 1 {
		t.Fatalf("got %d attempts, want 1 (non-retryable error must not retry)", attempts)
	}
}

func TestCircuitBreakerProvider_CallExhaustsRetriesAndReturnsLastError(t *testing.T) {
	inner := NewMockModelProvider()
	attempts := 0
	inner.SetCallFunc(func(ctx context.Context, prompt Prompt) (Response, error) {
		attempts++
		return Response{}, context.DeadlineExceeded
	})

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 2, BackoffFunc: noBackoff},
	})

	_, err := p.Call(context.Background(), Prompt{})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got err %v, want context.DeadlineExceeded", err)
	}
	if attempts != 3 { // initial attempt + 2 retries
		t.Fatalf("got %d attempts, want 3", attempts)
	}
}

func TestCircuitBreakerProvider_OpenBreakerShortCircuitsWithoutCallingInner(t *testing.T) {
	inner := NewMockModelProvider()
	inner.SetCallFunc(func(ctx context.Context, prompt Prompt) (Response, error) {
		t.Fatal("inner provider must not be called while breaker is open")
		return Response{}, nil
	})

	breaker := &fakeBreaker{open: true}
	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Breaker: breaker,
		Retry:   RetryPolicy{MaxRetries: 2, BackoffFunc: noBackoff, IsRetryable: func(error) bool { return false }},
	})

	_, err := p.Call(context.Background(), Prompt{})
	if err == nil {
		t.Fatal("expected error from open breaker, got nil")
	}
	if breaker.calls != 1 {
		t.Fatalf("got %d breaker calls, want 1", breaker.calls)
	}
}

func TestCircuitBreakerProvider_StreamRetriesConnectionSetupOnly(t *testing.T) {
	inner := NewMockModelProvider()
	attempts := 0
	wantChan := make(chan Token, 1)
	wantChan <- Token{Content: "tok"}
	close(wantChan)

	inner.SetStreamFunc(func(ctx context.Context, prompt Prompt) (<-chan Token, error) {
		attempts++
		if attempts < 2 {
			return nil, context.DeadlineExceeded
		}
		return wantChan, nil
	})

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 2, BackoffFunc: noBackoff},
	})

	out, err := p.Stream(context.Background(), Prompt{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("got %d attempts, want 2", attempts)
	}
	tok, ok := <-out
	if !ok || tok.Content != "tok" {
		t.Fatalf("got token %+v ok=%v, want Content=%q ok=true", tok, ok, "tok")
	}
}

func TestCircuitBreakerProvider_ContextCancelledDuringBackoffAbortsRetry(t *testing.T) {
	inner := NewMockModelProvider()
	attempts := 0
	inner.SetCallFunc(func(ctx context.Context, prompt Prompt) (Response, error) {
		attempts++
		return Response{}, context.DeadlineExceeded
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the first backoff sleep

	p := NewCircuitBreakerProvider(inner, CircuitBreakerProviderConfig{
		Retry: RetryPolicy{MaxRetries: 5, BackoffFunc: func(int) time.Duration { return time.Hour }},
	})

	_, err := p.Call(ctx, Prompt{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got err %v, want context.Canceled", err)
	}
	if attempts != 1 {
		t.Fatalf("got %d attempts, want 1 (must abort at first backoff, not hang an hour)", attempts)
	}
}

func TestDefaultIsRetryable(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"deadline exceeded", context.DeadlineExceeded, true},
		{"context canceled", context.Canceled, true},
		{"net timeout", &net.DNSError{IsTimeout: true}, true},
		{"generic error", errors.New("boom"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := DefaultIsRetryable(tc.err); got != tc.want {
				t.Errorf("DefaultIsRetryable(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

func TestProviderFactory_CreateProvider_WrapsWithCircuitBreakerOnlyWhenConfigured(t *testing.T) {
	// Zero-value CircuitBreaker/RetryPolicy fields (every existing caller,
	// today) must not change the returned provider's type — no wrap.
	f := NewProviderFactory()
	p, err := f.CreateProvider(ProviderConfig{Type: ProviderTypeMock})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, wrapped := p.(*CircuitBreakerProvider); wrapped {
		t.Fatal("provider should not be wrapped when CircuitBreaker/RetryPolicy are unset")
	}

	p2, err := f.CreateProvider(ProviderConfig{Type: ProviderTypeMock, RetryPolicy: &RetryPolicy{MaxRetries: 1}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, wrapped := p2.(*CircuitBreakerProvider); !wrapped {
		t.Fatal("provider should be wrapped when RetryPolicy is set")
	}
}
