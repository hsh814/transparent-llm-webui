(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const folderState = new Map();
  const readyForms = new WeakSet();
  let toastTimer;
  let followLatest = true;
  let observedList;
  let observer;
  let previousSurface;

  function toast(message, error = false) {
    const box = $('#toast');
    box.textContent = message;
    box.classList.toggle('error', error);
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.hidden = true; }, error ? 8000 : 2400);
  }

  function storage(method, key, value) {
    try { return localStorage[method](key, value); } catch (_) { return null; }
  }
  function draftKey(form) { return `transparent:draft:${form.dataset.sessionId}`; }
  function resize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, Math.min(240, innerHeight * .3)) + 'px';
  }
  function closeFolderMenus() {
    $$('.folder-menu.open').forEach(menu => {
      menu.classList.remove('open');
      $('.folder-menu-btn', menu.parentElement).setAttribute('aria-expanded', 'false');
    });
  }
  function positionFolderMenu(menu, button) {
    const anchor = button.getBoundingClientRect();
    const sidebar = $('#sidebar');
    // On mobile, the transformed drawer is the fixed element's containing block.
    const origin = getComputedStyle(sidebar).transform === 'none' ? {left: 0, top: 0} : sidebar.getBoundingClientRect();
    const below = anchor.bottom + 4;
    const top = below + menu.offsetHeight <= innerHeight - 8 ? below : anchor.top - menu.offsetHeight - 4;
    menu.style.top = Math.max(8, top) - origin.top + 'px';
    menu.style.left = Math.max(8, Math.min(anchor.right - menu.offsetWidth, innerWidth - menu.offsetWidth - 8)) - origin.left + 'px';
  }
  function drawer(open) {
    $('.layout')?.classList.toggle('nav-open', open);
    $('.menu-btn')?.setAttribute('aria-expanded', String(open));
    if ($('.main')) $('.main').inert = open;
    if (matchMedia('(max-width: 720px)').matches && $('#sidebar')) $('#sidebar').inert = !open;
    if (open) $('.new-conversation')?.focus({preventScroll: true});
    else $('.menu-btn')?.focus({preventScroll: true});
  }
  function scrollBottom() {
    const list = $('#message-list');
    if (list) list.scrollTop = list.scrollHeight;
  }
  function syncMessages() {
    const list = $('#message-list');
    if (!list) return;
    if ($('.message', list)) $('.conversation-empty', list)?.remove();
    // htmx briefly retains old CSS classes while a same-ID element settles.
    // Generation state must use the new data attribute, not the transient class.
    const pending = $$('[data-generation-status="queued"], [data-generation-status="running"]', list);
    const composer = $('form.composer');
    if (composer) {
      $('button[type="submit"]', composer).disabled = pending.length > 0 || composer.classList.contains('htmx-request');
      $('.btn-stop', composer).hidden = !pending.length;
      $('.composer-status', composer).textContent = pending.length ? (pending.length > 1 ? `${pending.length} chunks remaining` : 'Generating response…') : '';
    }
    const deletes = $$('.delete-message', list);
    const lastMessage = $$('.message[id^="message-"]', list).reduce((latest, el) => Math.max(latest, Number(el.id.slice(8))), 0);
    deletes.forEach(button => { button.hidden = !!pending.length || button.closest('.message').id !== `message-${lastMessage}`; });
    $$('.streaming .reasoning', list).forEach(el => {
      const hidden = !$('.reasoning-text', el).textContent;
      if (el.hidden !== hidden) el.hidden = hidden;
    });
    pending.forEach(el => {
      if ($('.content', el)?.textContent || $('.reasoning-text', el)?.textContent) {
        const label = $('.generation-label', el);
        if (label.textContent !== 'Responding…') label.textContent = 'Responding…';
      }
    });
    if (followLatest) scrollBottom();
    const jump = $('.jump-latest');
    if (jump) jump.hidden = followLatest;
  }
  function filterChats() {
    const query = ($('#chat-search')?.value || '').trim().toLowerCase();
    $$('.folder-item').forEach(folder => {
      const folderMatches = $('.folder-name', folder).textContent.toLowerCase().includes(query);
      let matches = 0;
      $$('.session-item', folder).forEach(item => {
        const match = !query || folderMatches || $('.session-title', item).textContent.toLowerCase().includes(query);
        item.hidden = !match;
        if (match) matches++;
      });
      folder.hidden = !!query && !folderMatches && !matches;
      folder.classList.toggle('searching', !!query);
      const state = folderState.get(folder.dataset.folderId);
      const open = query ? !folder.hidden : (state ? state.open : folder.classList.contains('active'));
      folder.classList.toggle('is-open', open);
      folder.classList.toggle('is-all', !!state?.all);
      const showAll = $('.show-all', folder);
      if (showAll) showAll.textContent = state?.all ? 'Show less' : `Show all (${showAll.dataset.count})`;
      $('.folder-toggle', folder).setAttribute('aria-expanded', String(open));
    });
  }
  function initialize() {
    const surface = $('#chat-surface');
    const form = $('form.composer');
    if (form && !readyForms.has(form)) {
      readyForms.add(form);
      const textarea = $('textarea', form);
      textarea.value = storage('getItem', draftKey(form)) || '';
      resize(textarea);
    }
    const list = $('#message-list');
    if (list && list !== observedList) {
      observer?.disconnect();
      observedList = list;
      followLatest = true;
      list.addEventListener('scroll', () => {
        followLatest = list.scrollHeight - list.scrollTop - list.clientHeight < 90;
        const jump = $('.jump-latest');
        if (jump) jump.hidden = followLatest;
      }, {passive: true});
      let scheduled = false;
      observer = new MutationObserver(() => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => { scheduled = false; syncMessages(); });
      });
      observer.observe(list, {childList: true, subtree: true});
      requestAnimationFrame(scrollBottom);
    }
    if (surface && previousSurface !== surface) {
      const wasOpen = $('.layout')?.classList.contains('nav-open');
      if (wasOpen) drawer(false);
      previousSurface = surface;
      const id = $('#active-session')?.value;
      history.replaceState(history.state, '', id ? `/?session=${id}` : '/');
    }
    const title = $('.title-input');
    const active = $('.session-item.active .session-title');
    if (title && active && document.activeElement !== title) title.value = active.textContent;
    $$('.session-modified').forEach(element => {
      if (element.dataset.formatted === element.dateTime) return;
      const date = new Date(element.dateTime);
      if (Number.isNaN(date.getTime())) return;
      element.textContent = 'Updated ' + new Intl.DateTimeFormat(undefined, {
        year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      }).format(date);
      element.title = 'Last modified: ' + new Intl.DateTimeFormat(undefined, {
        dateStyle: 'full', timeStyle: 'long',
      }).format(date);
      element.dataset.formatted = element.dateTime;
    });
    filterChats();
    syncMessages();
  }

  async function copy(text, button) {
    try {
      if (navigator.clipboard) await navigator.clipboard.writeText(text);
      else {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.cssText = 'position:fixed;top:-10000px';
        document.body.append(textarea);
        textarea.select();
        const copied = document.execCommand('copy');
        textarea.remove();
        if (!copied) throw new Error('Clipboard unavailable');
      }
      const label = button.textContent;
      button.textContent = 'Copied';
      setTimeout(() => { button.textContent = label; }, 1500);
    } catch (_) { toast('Could not access the clipboard. Select the text and copy it manually.', true); }
  }

  document.addEventListener('input', event => {
    if (event.target.matches('.composer textarea')) {
      const form = event.target.closest('form');
      storage('setItem', draftKey(form), event.target.value);
      resize(event.target);
    }
    if (event.target.id === 'chat-search') filterChats();
  });
  document.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault(); $('.new-conversation')?.click();
    }
    if (event.key === 'Escape') {
      if ($('.layout.nav-open')) drawer(false);
      const menu = document.activeElement?.closest('.folder-menu.open');
      if (menu) $('.folder-menu-btn', menu.parentElement).focus();
      closeFolderMenus();
      $$('.params-toggle[open]').forEach(details => { details.open = false; });
    }
    if (event.target.matches('.composer textarea') && event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      const form = event.target.closest('form');
      if (form.dataset.mode !== 'chat' && !event.metaKey && !event.ctrlKey) return;
      if (matchMedia('(pointer: coarse)').matches && !event.metaKey && !event.ctrlKey) return;
      event.preventDefault();
      if (!$('button[type="submit"]', form).disabled && event.target.value.trim()) form.requestSubmit();
    }
  });
  document.addEventListener('click', event => {
    const target = event.target;
    if (!target.closest('.params-toggle')) $$('.params-toggle[open]').forEach(details => { details.open = false; });
    if (target.closest('.menu-btn')) drawer(!$('.layout').classList.contains('nav-open'));
    if (target.closest('.scrim')) drawer(false);
    if (target.closest('.jump-latest')) { followLatest = true; scrollBottom(); syncMessages(); }
    const toggle = target.closest('.folder-toggle');
    if (toggle) {
      const folder = toggle.closest('.folder-item');
      const open = folder.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
      folderState.set(folder.dataset.folderId, {open, all: folder.classList.contains('is-all')});
    }
    const showAll = target.closest('.show-all');
    if (showAll) {
      const folder = showAll.closest('.folder-item');
      const all = folder.classList.toggle('is-all');
      folderState.set(folder.dataset.folderId, {open: true, all});
      showAll.textContent = all ? 'Show less' : `Show all (${showAll.dataset.count})`;
    }
    const menuButton = target.closest('.folder-menu-btn');
    $$('.folder-menu').forEach(menu => {
      const button = $('.folder-menu-btn', menu.parentElement);
      const open = button === menuButton && !menu.classList.contains('open');
      menu.classList.toggle('open', open);
      button.setAttribute('aria-expanded', String(open));
      if (open) positionFolderMenu(menu, button);
    });
    const copyButton = target.closest('.btn-copy');
    if (copyButton) copy(JSON.parse(copyButton.dataset.copy), copyButton);
    const copyAll = target.closest('#copy-all-btn');
    if (copyAll) copy($$('#message-list .message, .prompt-messages .message').map(el => `${$('.role-label', el)?.textContent || ''}\n${$('.content', el)?.textContent || ''}`).join('\n\n'), copyAll);
  });
  document.body.addEventListener('htmx:beforeRequest', event => {
    const form = event.detail.elt;
    const title = $('.title-input', form);
    if (title) form._submittedTitle = title.value;
    if (form.matches('form.composer')) {
      if ($('.streaming') || !$('textarea', form).value.trim()) { event.preventDefault(); return; }
      form._submittedDraft = $('textarea', form).value;
      followLatest = true;
    }
  });
  document.body.addEventListener('htmx:afterRequest', event => {
    const form = event.detail.elt;
    const title = $('.title-input', form);
    const savedTitle = $('.session-item.active .session-title');
    if (event.detail.successful && title && savedTitle && title.value === form._submittedTitle) {
      title.value = savedTitle.textContent;
    }
    if (event.detail.successful && form.matches('form.composer')) {
      const textarea = $('textarea', form);
      if (textarea.value === form._submittedDraft) {
        textarea.value = ''; storage('removeItem', draftKey(form)); resize(textarea);
      }
      textarea.focus({preventScroll: true});
    }
    syncMessages();
  });
  ['htmx:afterSwap', 'htmx:afterSettle', 'htmx:oobAfterSwap', 'htmx:historyRestore'].forEach(name => document.body.addEventListener(name, () => requestAnimationFrame(initialize)));
  document.body.addEventListener('htmx:responseError', event => {
    let message = 'The request failed. Your draft is saved. Please try again.';
    try {
      const detail = JSON.parse(event.detail.xhr.responseText).detail;
      message = typeof detail === 'string' ? detail : detail.map(error => `${error.loc.at(-1)}: ${error.msg}`).join(' · ');
    } catch (_) { /* Non-JSON errors use the safe fallback. */ }
    toast(message, true);
  });
  ['htmx:sendError', 'htmx:timeout'].forEach(name => document.body.addEventListener(name, () => toast('Connection lost. Your draft is saved. Please try again.', true)));
  document.body.addEventListener('htmx:sseError', () => {
    const status = $('.composer-status');
    if (status) status.textContent = 'Reconnecting… Your response is saved on the server.';
  });
  document.body.addEventListener('settingsSaved', () => {
    $$('.params-toggle[open]').forEach(details => { details.open = false; });
    toast('Settings saved for your next message.');
  });
  addEventListener('resize', () => {
    closeFolderMenus();
    const mobile = matchMedia('(max-width: 720px)').matches;
    if (!mobile) {
      $('.layout')?.classList.remove('nav-open');
      $('.menu-btn')?.setAttribute('aria-expanded', 'false');
      if ($('.main')) $('.main').inert = false;
    }
    if ($('#sidebar')) $('#sidebar').inert = mobile && !$('.layout').classList.contains('nav-open');
  });
  document.addEventListener('scroll', event => {
    if (event.target.matches?.('#folder-list, .sidebar')) closeFolderMenus();
  }, true);
  initialize();
  dispatchEvent(new Event('resize'));
})();
