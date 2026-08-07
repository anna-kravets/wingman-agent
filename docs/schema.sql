-- Supabase schema for multi-turn conversation state.
-- Run once against the Supabase project (Phase 1).

create table if not exists conversations (
  conversation_id text primary key,
  history jsonb not null default '[]'::jsonb,
  updated_at timestamptz not null default now()
);
