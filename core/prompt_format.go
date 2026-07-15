package core

import (
	"bytes"
	"fmt"
	"regexp"
	"text/template"
)

// FormatType selects the substitution syntax used by FormatPromptString.
// Modeled after cloudwego/eino's schema.FormatType (FString/GoTemplate/Jinja2),
// scoped down to the two syntaxes implementable without a new dependency.
type FormatType uint8

const (
	// FormatGoTemplate interpolates using Go's text/template ({{.key}}).
	// A string with no template directives passes through unchanged, so this
	// is safe to apply unconditionally to existing plain-text system prompts.
	FormatGoTemplate FormatType = iota
	// FormatFString interpolates using Python str.format-style {key} placeholders.
	FormatFString
)

var fstringPlaceholder = regexp.MustCompile(`\{([a-zA-Z_][a-zA-Z0-9_]*)\}`)

// FormatPromptString substitutes vs into tpl according to formatType.
//
// Variable keys present in tpl but absent from vs produce an error for
// FormatFString, and render as "<no value>" for FormatGoTemplate (the
// text/template default) — there is no compile-time safety, consistent with
// eino's ChatTemplate.Format contract.
func FormatPromptString(tpl string, vs map[string]any, formatType FormatType) (string, error) {
	switch formatType {
	case FormatFString:
		return formatFString(tpl, vs)
	case FormatGoTemplate:
		return formatGoTemplate(tpl, vs)
	default:
		return "", fmt.Errorf("core: unsupported FormatType %d", formatType)
	}
}

func formatGoTemplate(tpl string, vs map[string]any) (string, error) {
	t, err := template.New("prompt").Parse(tpl)
	if err != nil {
		return "", fmt.Errorf("core: parse go template: %w", err)
	}
	var buf bytes.Buffer
	if err := t.Execute(&buf, vs); err != nil {
		return "", fmt.Errorf("core: execute go template: %w", err)
	}
	return buf.String(), nil
}

func formatFString(tpl string, vs map[string]any) (string, error) {
	var missing error
	result := fstringPlaceholder.ReplaceAllStringFunc(tpl, func(match string) string {
		key := match[1 : len(match)-1]
		v, ok := vs[key]
		if !ok {
			if missing == nil {
				missing = fmt.Errorf("core: missing variable %q for fstring template", key)
			}
			return match
		}
		return fmt.Sprintf("%v", v)
	})
	if missing != nil {
		return "", missing
	}
	return result, nil
}

// StateVars snapshots a State's data map into a plain map[string]any suitable
// for FormatPromptString. State does not expose its underlying map directly,
// so this walks Keys()+Get().
func StateVars(state State) map[string]any {
	keys := state.Keys()
	vs := make(map[string]any, len(keys))
	for _, k := range keys {
		if v, ok := state.Get(k); ok {
			vs[k] = v
		}
	}
	return vs
}
