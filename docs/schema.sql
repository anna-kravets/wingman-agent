-- Supabase schema for anonymous, device-scoped multi-turn conversation state.
-- Use this for a fresh project. Existing projects should run the migration in
-- supabase/migrations/202608120001_anonymous_conversation_owners.sql instead.

create table if not exists conversations (
  owner_id text not null,
  conversation_id text not null,
  title text not null default 'New conversation',
  history jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (owner_id, conversation_id)
);

create index if not exists conversations_owner_updated_idx
  on conversations (owner_id, updated_at desc);

-- The FastAPI server uses the service-role key. With no public policies,
-- browsers cannot query conversation rows directly through Supabase REST.
alter table public.conversations enable row level security;
