-- Run once in Supabase Dashboard > SQL Editor.
-- Every table is private to the authenticated owner; never use a service-role key in the app.

create table if not exists public.climate_chats (
    id text primary key,
    user_id uuid not null references auth.users(id) on delete cascade,
    title text not null default 'New chat',
    title_is_custom boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Existing projects can rerun this file safely to add the manual-rename marker.
alter table public.climate_chats
    add column if not exists title_is_custom boolean not null default false;

create table if not exists public.climate_messages (
    id text primary key,
    chat_id text not null references public.climate_chats(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

-- Experiment outputs stay in their own table. Chat messages store only an experiment_id pointer.
create table if not exists public.climate_experiments (
    id text primary key,
    chat_id text not null references public.climate_chats(id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    record jsonb not null,
    report text not null default '',
    created_at timestamptz not null default now()
);

create index if not exists climate_chats_owner_recent_idx
    on public.climate_chats (user_id, updated_at desc);
create index if not exists climate_messages_chat_order_idx
    on public.climate_messages (user_id, chat_id, created_at);
create index if not exists climate_experiments_chat_idx
    on public.climate_experiments (user_id, chat_id, created_at);

alter table public.climate_chats enable row level security;
alter table public.climate_messages enable row level security;
alter table public.climate_experiments enable row level security;

revoke all on public.climate_chats from public, anon;
revoke all on public.climate_messages from public, anon;
revoke all on public.climate_experiments from public, anon;
grant select, insert, update, delete on public.climate_chats to authenticated;
grant select, insert, update, delete on public.climate_messages to authenticated;
grant select, insert, update, delete on public.climate_experiments to authenticated;

drop policy if exists climate_chats_select_own on public.climate_chats;
create policy climate_chats_select_own on public.climate_chats
    for select to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy if exists climate_chats_insert_own on public.climate_chats;
create policy climate_chats_insert_own on public.climate_chats
    for insert to authenticated
    with check ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy if exists climate_chats_update_own on public.climate_chats;
create policy climate_chats_update_own on public.climate_chats
    for update to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()))
    with check ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy if exists climate_chats_delete_own on public.climate_chats;
create policy climate_chats_delete_own on public.climate_chats
    for delete to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));

drop policy if exists climate_messages_select_own on public.climate_messages;
create policy climate_messages_select_own on public.climate_messages
    for select to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy if exists climate_messages_insert_own on public.climate_messages;
create policy climate_messages_insert_own on public.climate_messages
    for insert to authenticated
    with check (
        (select auth.uid()) is not null
        and user_id = (select auth.uid())
        and exists (
            select 1 from public.climate_chats c
            where c.id = chat_id and c.user_id = (select auth.uid())
        )
    );
drop policy if exists climate_messages_update_own on public.climate_messages;
create policy climate_messages_update_own on public.climate_messages
    for update to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()))
    with check (
        (select auth.uid()) is not null
        and user_id = (select auth.uid())
        and exists (
            select 1 from public.climate_chats c
            where c.id = chat_id and c.user_id = (select auth.uid())
        )
    );
drop policy if exists climate_messages_delete_own on public.climate_messages;
create policy climate_messages_delete_own on public.climate_messages
    for delete to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));

drop policy if exists climate_experiments_select_own on public.climate_experiments;
create policy climate_experiments_select_own on public.climate_experiments
    for select to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));
drop policy if exists climate_experiments_insert_own on public.climate_experiments;
create policy climate_experiments_insert_own on public.climate_experiments
    for insert to authenticated
    with check (
        (select auth.uid()) is not null
        and user_id = (select auth.uid())
        and exists (
            select 1 from public.climate_chats c
            where c.id = chat_id and c.user_id = (select auth.uid())
        )
    );
drop policy if exists climate_experiments_update_own on public.climate_experiments;
create policy climate_experiments_update_own on public.climate_experiments
    for update to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()))
    with check (
        (select auth.uid()) is not null
        and user_id = (select auth.uid())
        and exists (
            select 1 from public.climate_chats c
            where c.id = chat_id and c.user_id = (select auth.uid())
        )
    );
drop policy if exists climate_experiments_delete_own on public.climate_experiments;
create policy climate_experiments_delete_own on public.climate_experiments
    for delete to authenticated
    using ((select auth.uid()) is not null and user_id = (select auth.uid()));
