# Supabase production handoff — Anna

This is the remaining database work for anonymous, no-login conversation history. The application
code and tests are ready. Do the database migration **before** deploying the matching application
version so conversation restore/delete becomes available immediately.

## 1. Confirm the target

In the Supabase dashboard, open the project whose URL equals the production `SUPABASE_URL` in
Vercel. In **Table Editor**, confirm that `public.conversations` exists. Before changing it, run this
in **SQL Editor** and save the result:

```sql
select count(*) as existing_conversations from public.conversations;

select column_name, data_type, is_nullable
from information_schema.columns
where table_schema = 'public' and table_name = 'conversations'
order by ordinal_position;
```

Expected old columns include `conversation_id`, `history`, and `updated_at`. Do not create a second
table named `Conversations`; PostgreSQL names are case-sensitive when quoted, and the code uses the
lowercase `public.conversations` table.

## 2. Apply the reviewed migration

Open and copy the complete contents of:

```text
supabase/migrations/202608120001_anonymous_conversation_owners.sql
```

Paste it into a new SQL Editor query and click **Run** once. The script:

- preserves every existing `history` value;
- adds `owner_id`, `title`, `created_at`, and `updated_at`;
- assigns old rows an unreachable `legacy-<conversation_id>` owner so they are not exposed to a
  new anonymous device;
- changes the primary key to `(owner_id, conversation_id)`;
- adds an index for each device's recent-conversation list; and
- enables Row Level Security with no public policies.

## 3. Verify the live schema

Run:

```sql
select column_name, data_type, is_nullable
from information_schema.columns
where table_schema = 'public' and table_name = 'conversations'
order by ordinal_position;

select a.attname as primary_key_column
from pg_index i
cross join lateral unnest(i.indkey) with ordinality as key_column(attnum, position)
join pg_attribute a on a.attrelid = i.indrelid and a.attnum = key_column.attnum
where i.indrelid = 'public.conversations'::regclass
  and i.indisprimary
order by key_column.position;

select relrowsecurity
from pg_class
where oid = 'public.conversations'::regclass;

select count(*) as rows_missing_owner
from public.conversations
where owner_id is null;
```

Expected results:

- all seven columns exist: `owner_id`, `conversation_id`, `title`, `history`, `created_at`,
  `updated_at` (plus any team-owned columns already present);
- primary-key columns are `owner_id`, then `conversation_id`;
- `relrowsecurity` is `true`;
- `rows_missing_owner` is `0`.

## 4. Configure Vercel secrets

In the Vercel project, add these to **Production**, **Preview**, and **Development** as appropriate:

```text
SUPABASE_URL=<this Supabase project URL>
SUPABASE_SERVICE_ROLE_KEY=<service_role key from Supabase API settings>
```

Important:

- Use the **service_role** key, not the public anon/publishable key.
- Treat it as a secret; never paste it into source code, GitHub, `public/index.html`, or any variable
  whose name begins with `NEXT_PUBLIC_`.
- The repository keeps `SUPABASE_KEY` only as a compatibility fallback. Production should use
  `SUPABASE_SERVICE_ROLE_KEY`.

Redeploy the application after saving the variables.

## 5. No-cost production smoke test

These checks do not invoke the agent or any paid model. Replace `<APP_URL>` with the production URL:

```powershell
$wingmanSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession

$first = Invoke-RestMethod `
  -Uri "<APP_URL>/api/conversations" `
  -WebSession $wingmanSession

$second = Invoke-RestMethod `
  -Uri "<APP_URL>/api/conversations" `
  -WebSession $wingmanSession

$delete = Invoke-RestMethod `
  -Method Delete `
  -Uri "<APP_URL>/api/conversations/migration-smoke-test-does-not-exist" `
  -WebSession $wingmanSession

$first
$second
$delete
```

Expected: both list calls return `status: ok` with a `conversations` array, and deletion returns
`status: ok`. The same web session sends the same anonymous cookie. Deleting the nonexistent test ID
does not alter data.

Finally, open the production UI, confirm there is no demo conversation/button, and check that the
delete button opens the confirmation dialog. Do not send an agent prompt merely to test the UI.

## 6. Handoff result

Report back with:

- confirmation that the migration ran successfully;
- the four verification-query results (do not include secret values);
- confirmation that `SUPABASE_SERVICE_ROLE_KEY` is set in Vercel; and
- the three smoke-test statuses.

Do not remove the legacy rows during this task. They may contain earlier test history and can be
reviewed separately after the production migration is confirmed.

## 7. If something fails

Do not drop the new columns or delete conversation rows. The application deliberately continues in
single-turn mode when Supabase history is unavailable, so database trouble does not block passenger
answers. Capture the SQL error and stop the rollout.

If the application version must be rolled back, roll back the application first and leave the added
columns in place; they are backward-compatible. Restoring the old `conversation_id`-only primary key
is unsafe after more than one anonymous owner has used the same conversation ID, so do not attempt
that without first checking for duplicates and coordinating with the team.
