-- Per-turn execution traces, for the GUI's "Execution details" panel.
--
-- Deliberately NOT a column on public.conversations. That row's `history` is read in
-- full on every /api/execute call before the agent runs, and one turn's trace can run
-- to a quarter of a megabyte -- a six-turn conversation would mean downloading and
-- parsing megabytes to feed code that never looks at a step. In its own table the
-- agent path never pays for it, and storing a turn is an insert rather than a
-- read-modify-write of a growing blob.
--
-- Traces were previously never persisted at all: they reached the browser in the
-- /api/execute response and lived only in localStorage, so any turn whose response
-- the browser missed (a reload, an abort, a slow run) lost its trace permanently once
-- the server copy was synced back over it.

create table if not exists public.conversation_traces (
  owner_id text not null,
  conversation_id text not null,
  turn_index integer not null,
  steps jsonb not null,
  created_at timestamptz not null default now(),
  primary key (owner_id, conversation_id, turn_index)
);

-- The primary key already indexes (owner_id, conversation_id) as its leading columns,
-- which is what the per-device listing filters on. No second index needed.

-- Same posture as public.conversations: the FastAPI server uses the service-role key,
-- which bypasses RLS. Do not add anon/authenticated policies -- all access must pass
-- through the owner-scoped API routes.
alter table public.conversation_traces enable row level security;
