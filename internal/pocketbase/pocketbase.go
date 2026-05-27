// Package pocketbase sets up the embedded PocketBase app with collections for
// API key management, token usage recording, and reasoning cache.
package pocketbase

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/pocketbase/dbx"
	"github.com/pocketbase/pocketbase"
	"github.com/pocketbase/pocketbase/core"

	"github.com/a876691666/deepseek-cursor-proxy/internal/config"
)

// Collection names.
const (
	CollectionAPIKeys        = "api_keys"
	CollectionTokenUsage     = "token_usage"
	CollectionReasoningCache = "reasoning_cache"
)

//nolint:unused
func init() {
	// These constants exist for external reference.
	_ = CollectionAPIKeys
	_ = CollectionTokenUsage
	_ = CollectionReasoningCache
}

// Setup creates, bootstraps, and initialises the PocketBase application.
func Setup(cfg config.Config) (*pocketbase.PocketBase, error) {
	pb := pocketbase.NewWithConfig(pocketbase.Config{
		DefaultDataDir:  cfg.PBDataDir,
		HideStartBanner: true,
	})

	if err := pb.Bootstrap(); err != nil {
		return nil, fmt.Errorf("bootstrap: %w", err)
	}

	for _, c := range allCollections() {
		if err := ensureCollection(pb, c); err != nil {
			return nil, err
		}
	}

	if err := ensureSuperuser(pb, cfg.PBAdminEmail, cfg.PBAdminPassword); err != nil {
		return nil, fmt.Errorf("superuser: %w", err)
	}

	return pb, nil
}

type collectionDef struct {
	name      string
	fields    core.FieldsList
	listRule  *string
	viewRule  *string
	createRule *string
	updateRule *string
	deleteRule *string
}

func allCollections() []collectionDef {
	superuserOnly := ns("@request.auth.isSuperuser = true")
	return []collectionDef{
		{
			name: CollectionAPIKeys,
			fields: core.FieldsList{
				&core.TextField{Name: "key", Required: true},
				&core.TextField{Name: "name", Required: true},
				&core.BoolField{Name: "active"},
			},
			listRule: superuserOnly, viewRule: superuserOnly,
			createRule: superuserOnly, updateRule: superuserOnly, deleteRule: superuserOnly,
		},
		{
			name: CollectionTokenUsage,
			fields: core.FieldsList{
				&core.NumberField{Name: "prompt_tokens"},
				&core.NumberField{Name: "completion_tokens"},
				&core.NumberField{Name: "total_tokens"},
				&core.TextField{Name: "api_key"},
				&core.TextField{Name: "model"},
				&core.TextField{Name: "recorded_at"},
			},
			listRule: superuserOnly, viewRule: superuserOnly,
			createRule: ns(""), updateRule: superuserOnly, deleteRule: superuserOnly,
		},
		{
			name: CollectionReasoningCache,
			fields: core.FieldsList{
				&core.TextField{Name: "key", Required: true},
				&core.TextField{Name: "reasoning"},
				&core.TextField{Name: "message_json"},
			},
			listRule: superuserOnly, viewRule: superuserOnly,
			createRule: ns(""), updateRule: ns(""), deleteRule: superuserOnly,
		},
	}
}

func ns(v string) *string {
	if v == "" {
		return nil
	}
	return &v
}

func ensureCollection(app core.App, def collectionDef) error {
	if _, err := app.FindCollectionByNameOrId(def.name); err == nil {
		return nil
	}
	c := core.NewBaseCollection(def.name)
	c.Fields = def.fields
	c.ListRule, c.ViewRule = def.listRule, def.viewRule
	c.CreateRule, c.UpdateRule = def.updateRule, def.createRule
	c.DeleteRule = def.deleteRule
	return app.Save(c)
}

func ensureSuperuser(app core.App, email, password string) error {
	users, err := app.FindAllRecords("_superusers", dbx.HashExp{"email": email})
	if err != nil {
		return fmt.Errorf("lookup: %w", err)
	}
	for _, u := range users {
		u.SetPassword(password)
		return app.Save(u)
	}
	c, err := app.FindCollectionByNameOrId("_superusers")
	if err != nil {
		return fmt.Errorf("_superusers: %w", err)
	}
	r := core.NewRecord(c)
	r.SetEmail(email)
	r.SetPassword(password)
	return app.Save(r)
}

// GenerateAPIKey returns a new random key with the sk-dcp- prefix.
func GenerateAPIKey() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "sk-dcp-" + hex.EncodeToString(b), nil
}

// CreateAPIKey inserts and returns a new API key record.
func CreateAPIKey(app core.App, name string) (*core.Record, error) {
	c, err := app.FindCollectionByNameOrId(CollectionAPIKeys)
	if err != nil {
		return nil, err
	}
	k, err := GenerateAPIKey()
	if err != nil {
		return nil, err
	}
	r := core.NewRecord(c)
	r.Set("key", k)
	r.Set("name", name)
	r.Set("active", true)
	if err := app.Save(r); err != nil {
		return nil, err
	}
	return r, nil
}

// LookupAPIKey finds an active API key record. Returns nil when not found/disabled.
func LookupAPIKey(app core.App, key string) (*core.Record, error) {
	r, err := app.FindFirstRecordByData(CollectionAPIKeys, "key", key)
	if err != nil || !r.GetBool("active") {
		return nil, nil
	}
	return r, nil
}

// RecordTokenUsage inserts a token usage record.
func RecordTokenUsage(app core.App, apiKey, model string, promptTokens, completionTokens, totalTokens int, recordedAt time.Time) error {
	c, err := app.FindCollectionByNameOrId(CollectionTokenUsage)
	if err != nil {
		return err
	}
	r := core.NewRecord(c)
	r.Set("api_key", apiKey)
	r.Set("model", model)
	r.Set("prompt_tokens", promptTokens)
	r.Set("completion_tokens", completionTokens)
	r.Set("total_tokens", totalTokens)
	r.Set("recorded_at", recordedAt.UTC().Format(time.RFC3339))
	return app.Save(r)
}
