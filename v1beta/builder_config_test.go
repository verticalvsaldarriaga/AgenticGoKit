package v1beta

import (
	"testing"
	"time"
)

// TestWithConfigPreservesBuilderName verifies that WithConfig does not
// discard the name passed to NewBuilder (issue #137, additional finding A).
func TestWithConfigPreservesBuilderName(t *testing.T) {
	b := NewBuilder("sales").WithConfig(&Config{
		LLM: LLMConfig{Provider: "mock", Model: "mock-model"},
	})

	sb, ok := b.(*streamlinedBuilder)
	if !ok {
		t.Fatalf("unexpected builder type %T", b)
	}
	if sb.config.Name != "sales" {
		t.Errorf("config.Name = %q, want %q from NewBuilder", sb.config.Name, "sales")
	}
	if sb.config.Timeout <= 0 {
		t.Errorf("config.Timeout = %v, want builder default preserved", sb.config.Timeout)
	}
}

// TestWithConfigExplicitValuesWin verifies user-provided values are not
// overwritten by builder defaults.
func TestWithConfigExplicitValuesWin(t *testing.T) {
	b := NewBuilder("ignored").WithConfig(&Config{
		Name:    "explicit",
		Timeout: 42 * time.Second,
		LLM:     LLMConfig{Provider: "mock", Model: "mock-model"},
	})

	sb := b.(*streamlinedBuilder)
	if sb.config.Name != "explicit" {
		t.Errorf("config.Name = %q, want %q", sb.config.Name, "explicit")
	}
	if sb.config.Timeout != 42*time.Second {
		t.Errorf("config.Timeout = %v, want 42s", sb.config.Timeout)
	}
}
