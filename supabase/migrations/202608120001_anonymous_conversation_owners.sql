-- Device-scoped conversation persistence for the no-login chat UI.
-- Existing rows are deliberately assigned an unreachable legacy owner rather
-- than exposed to the first browser that happens to reuse their conversation ID.

alter table public.conversations
  add column if not exists owner_id text,
  add column if not exists title text not null default 'New conversation',
  add column if not exists created_at timestamptz not null default now(),
  add column if not exists updated_at timestamptz not null default now();

update public.conversations
set owner_id = 'legacy-' || conversation_id
where owner_id is null;

alter table public.conversations
  alter column owner_id set not null;

alter table public.conversations
  drop constraint if exists conversations_pkey;

alter table public.conversations
  add constraint conversations_pkey primary key (owner_id, conversation_id);

create index if not exists conversations_owner_updated_idx
  on public.conversations (owner_id, updated_at desc);

-- The FastAPI server uses the service-role key. Do not add anon/authenticated
-- policies: all access must pass through the owner-scoped API routes.
alter table public.conversations enable row level security;
