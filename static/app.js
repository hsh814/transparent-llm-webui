(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const folderState = new Map();
  const readyForms = new WeakSet();
  // Files are deliberately memory-only; text and per-chat search settings survive reload.
  const attachmentDrafts = new Map();
  let toastTimer;
  let followLatest = true;
  let observedList;
  let observer;
  let previousSurface;
  let activeMessageRequest = null;

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
  function searchKey(form) { return `transparent:search:${form.dataset.sessionId}`; }
  function attachmentState(form) {
    const id = form.dataset.sessionId;
    if (!attachmentDrafts.has(id)) attachmentDrafts.set(id, []);
    return attachmentDrafts.get(id);
  }
  function hasContent(form) {
    return !!$('textarea', form).value.trim() || (form.dataset.mode === 'chat' && attachmentState(form).length > 0);
  }
  function syncInputSummary(form) {
    const summary = $('.input-summary', form);
    if (!summary) return;
    const count = attachmentState(form).length;
    const state = [];
    if (count) state.push(`${count} ${count === 1 ? 'file' : 'files'}`);
    if ($('[name="web_search"]', form).checked) state.push('Search allowed');
    summary.textContent = state.length ? `· ${state.join(' · ')}` : '';
    summary.hidden = state.length === 0;
  }
  function renderAttachments(form) {
    const list = $('.attachment-drafts', form);
    if (!list) return;
    const entries = attachmentState(form);
    list.replaceChildren();
    list.hidden = entries.length === 0;
    $('.file-draft-note', form).hidden = entries.length === 0;
    syncInputSummary(form);
    entries.forEach((entry, index) => {
      const item = document.createElement('li');
      if (entry.url) {
        const image = document.createElement('img');
        image.src = entry.url;
        image.alt = `Preview of ${entry.file.name}`;
        item.append(image);
      }
      const name = document.createElement('span');
      name.textContent = `${entry.file.name} · ${(entry.file.size / 1048576).toFixed(2)} MiB`;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'btn-mini remove-attachment';
      remove.dataset.attachmentIndex = String(index);
      remove.textContent = 'Remove';
      remove.setAttribute('aria-label', `Remove ${entry.file.name}`);
      item.append(name, remove);
      list.append(item);
    });
  }
  function saveSearch(form) {
    const checkbox = $('[name="web_search"]', form);
    if (!checkbox) return;
    storage('setItem', searchKey(form), JSON.stringify({enabled: checkbox.checked}));
    syncInputSummary(form);
  }
  function resize(textarea) {
    textarea.style.height = 'auto';
    if (!textarea.value) return;
    const style = getComputedStyle(textarea);
    const borders = parseFloat(style.borderTopWidth) + parseFloat(style.borderBottomWidth);
    textarea.style.height = Math.min(textarea.scrollHeight + borders, Math.min(240, innerHeight * .3)) + 'px';
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
      $('button[type="submit"]', composer).disabled = pending.length > 0 || !!activeMessageRequest || composer.classList.contains('htmx-request');
      $('.btn-stop', composer).hidden = !pending.length;
      $('.composer-status', composer).textContent = pending.length ? (pending.length > 1 ? `${pending.length} chunks remaining` : 'Generating response…') : '';
    }
    // Variants may have older database IDs than the preceding visible message.
    // Only displayed order determines the terminal turn.
    const saved = $$('.message[data-message-role]', list).filter(el => el.dataset.messageRole !== 'system');
    const lastMessage = saved.at(-1);
    const busy = pending.length > 0 || !!activeMessageRequest;
    $$('.open-batch-editor').forEach(button => { button.disabled = busy; });
    saved.forEach(message => {
      const terminal = message === lastMessage;
      message.dataset.terminal = String(terminal);
      $$('[data-terminal-action]', message).forEach(button => {
        button.hidden = !terminal || (button.matches('.request-response') && pending.length > 0);
      });
      const variants = $('.response-variants', message);
      if (variants) {
        variants.dataset.canSwitch = String(terminal && !busy);
        $('.variant-lock', variants).hidden = terminal;
        $$('[data-variant-action]', variants).forEach(button => {
          button.title = terminal ? button.getAttribute('aria-label') : 'Fork to change this response';
        });
      }
    });
    $$('[data-message-mutation]', list).forEach(button => {
      const terminal = button.closest('.message[data-message-role]') === lastMessage;
      button.disabled = busy || button.dataset.unavailable === 'true' || (button.hasAttribute('data-variant-action') && !terminal);
    });
    $$('.streaming .reasoning', list).forEach(el => {
      const hidden = !$('.reasoning-text', el).textContent;
      if (el.hidden !== hidden) el.hidden = hidden;
    });
    pending.forEach(el => {
      const progress = $('.generation-progress', el);
      if ($('.content', progress || el)?.textContent || $('.reasoning-text', progress || el)?.textContent) {
        const label = $('.generation-label', progress || el);
        if (label && label.textContent !== 'Responding…') label.textContent = 'Responding…';
      }
    });
    if (followLatest) scrollBottom();
    const jump = $('.jump-latest');
    if (jump) jump.hidden = followLatest;
  }
  function editorDraft(form) {
    const snapshot = form._snapshot;
    return {
      revision: snapshot.revision,
      messages: snapshot.messages.map(message => {
        const row = $(`[data-message-id="${message.id}"]`, form);
        return {id: message.id, include: $('.batch-include-message', row).checked,
          content: $('.batch-content', row).value,
          reasoning: $('.batch-reasoning', row)?.value ?? '',
          attachments: $$('.batch-attachment:checked', row).map(input => Number(input.dataset.index))};
      }),
      tools: snapshot.tool_groups.map(group => {
        const row = $$('.batch-tool-row', form).find(element => element.dataset.toolKey === group.key);
        return {key: group.key, include: $('.batch-include-tool', row).checked,
          reasoning: $('.batch-tool-reasoning', row).value, explicit: row._explicitChoice !== undefined};
      }),
    };
  }
  function updateEditor(form) {
    if (!form._snapshot || form._busy) return;
    const snapshot = form._snapshot;
    let firstChange = Infinity;
    let changed = 0;
    let excluded = 0;
    const states = snapshot.messages.map(message => {
      const row = $(`[data-message-id="${message.id}"]`, form);
      const include = $('.batch-include-message', row).checked;
      const attachments = $$('.batch-attachment:checked', row).map(input => Number(input.dataset.index));
      const edited = include && ($('.batch-content', row).value !== message.content
        || ($('.batch-reasoning', row)?.value ?? '') !== message.reasoning
        || attachments.join(',') !== message.attachments.map(attachment => attachment.index).join(','));
      if (!include || edited) firstChange = Math.min(firstChange, message.position);
      if (edited) changed++;
      if (!include) excluded++;
      row.classList.toggle('batch-excluded', !include);
      $('.batch-content', row).disabled = !include;
      if ($('.batch-reasoning', row)) $('.batch-reasoning', row).disabled = !include;
      $$('.batch-attachment', row).forEach(input => { input.disabled = !include; });
      return {message, row, include, edited};
    });
    const includedUsers = new Set(states.filter(state => state.include && state.message.role === 'user').map(state => state.message.id));
    const groups = snapshot.tool_groups.map(group => {
      const row = $$('.batch-tool-row', form).find(element => element.dataset.toolKey === group.key);
      const available = group.complete && includedUsers.has(group.user_message_id);
      const edited = $('.batch-tool-reasoning', row).value !== group.reasoning;
      if (!available || edited || row._explicitChoice === false) firstChange = Math.min(firstChange, group.position);
      return {group, row, available, edited};
    });
    let keptTools = 0;
    groups.sort((a, b) => a.group.position - b.group.position).forEach(({group, row, available, edited}) => {
      const input = $('.batch-include-tool', row);
      input.disabled = !available;
      $('.batch-tool-reasoning', row).disabled = !available;
      input.checked = available && (row._explicitChoice ?? (group.position < firstChange));
      if (!input.checked) firstChange = Math.min(firstChange, group.position);
      else keptTools++;
      row.classList.toggle('batch-excluded', !input.checked);
      $('.batch-tool-state', row).textContent = !group.complete ? 'Incomplete — cannot reuse'
        : !available ? 'Anchor user excluded' : input.checked ? (edited ? 'Manual Thinking · explicit reuse · not rerun' : row._explicitChoice === true ? 'Explicit reuse · not rerun' : 'Unchanged history · not rerun')
        : 'Excluded' + (row._explicitChoice === undefined ? ' after context change' : ' by choice');
    });
    let carried = 0;
    const preview = $('#batch-preview-list', form);
    preview.replaceChildren();
    const previewItems = [];
    states.forEach(({message, row, include, edited}) => {
      const isCarried = include && !edited && message.role === 'assistant' && message.position > firstChange;
      if (isCarried) carried++;
      const label = !include ? 'Excluded' : edited ? (message.role === 'assistant' ? 'Manual response' : 'Edited input')
        : isCarried ? 'Carried from earlier context' : 'Unchanged';
      $('.batch-row-state', row).textContent = label;
      if (!include) return;
      const item = document.createElement('li');
      const content = $('.batch-content', row).value;
      const reasoning = $('.batch-reasoning', row)?.value ?? '';
      item.textContent = `${message.role} · ${label}: ${content.slice(0, 140) || (reasoning.trim() ? '[Thinking-only response]' : message.role === 'user' ? '[attachment-only input]' : '[empty text]')}${reasoning ? ` · Thinking: ${reasoning.slice(0, 140)}` : ''}`;
      previewItems.push({position: message.position, item});
    });
    groups.forEach(({group, row, edited}) => {
      if (!$('.batch-include-tool', row).checked) return;
      const item = document.createElement('li');
      const reasoning = $('.batch-tool-reasoning', row).value;
      item.textContent = `Tool batch · ${edited ? 'Manual Thinking · explicit reuse' : 'Reused history'} · calls/results unchanged${reasoning ? ` · Thinking: ${reasoning.slice(0, 140)}` : ' · no Thinking'}`;
      previewItems.push({position: group.position, item});
    });
    previewItems.sort((a, b) => a.position - b.position).forEach(({item}) => preview.append(item));
    const kept = states.filter(state => state.include);
    const canSend = form.dataset.mode !== 'memo' && kept.at(-1)?.message.role === 'user';
    $('#batch-editor-send', form).disabled = !canSend;
    $('#batch-send-note', form).textContent = canSend
      ? 'Save & send queues only the final included user request. Earlier answers and reused tools are not rerun.'
      : form.dataset.mode === 'memo' ? 'Memo only saves notes; it never calls a model.'
      : 'Save as fork is available. To send, the last included message must be a user request.';
    const changedTools = groups.filter(({edited}) => edited).length;
    $('#batch-editor-summary', form).textContent = `${kept.length} messages kept · ${changed} edited · ${excluded} excluded · ${carried} carried answers · ${keptTools}/${groups.length} tool batches reused · ${changedTools} tool Thinking edited. Saving creates one independent fork.`;
    $('[name="draft"]', form).value = JSON.stringify(editorDraft(form));
  }
  function closeEditor(dialog) {
    if ($('#batch-editor-form', dialog)?._busy) return;
    const opener = dialog._opener;
    dialog.close();
    dialog.remove();
    if (opener?.isConnected) opener.focus({preventScroll: true});
  }
  function initializeEditor() {
    const dialog = $('#batch-editor');
    if (!dialog || dialog.open) return;
    const form = $('#batch-editor-form', dialog);
    form._snapshot = JSON.parse($('#batch-editor-snapshot', dialog).textContent);
    dialog._opener = document.activeElement;
    dialog.addEventListener('cancel', event => { event.preventDefault(); closeEditor(dialog); });
    dialog.addEventListener('input', event => {
      if (event.target.matches('.batch-reasoning, .batch-tool-reasoning')) {
        const position = Number(event.target.closest('[data-position]').dataset.position);
        $$('.batch-tool-row', form).forEach(row => {
          if (Number(row.dataset.position) >= position) row._explicitChoice = undefined;
        });
      }
      if (event.target.matches('.batch-include-tool')) event.target.closest('.batch-tool-row')._explicitChoice = event.target.checked;
      updateEditor(form);
    });
    dialog.addEventListener('change', event => {
      if (event.target.matches('.batch-include-tool')) event.target.closest('.batch-tool-row')._explicitChoice = event.target.checked;
      updateEditor(form);
    });
    updateEditor(form);
    dialog.showModal();
    const focus = dialog.dataset.focusMessage;
    const toolRow = dialog.dataset.focusTool
      ? $$('.batch-tool-row', dialog).find(row => row.dataset.toolKey === dialog.dataset.focusTool) : null;
    const textarea = toolRow ? $('.batch-tool-reasoning', toolRow)
      : focus ? $(`#batch-content-${focus}`, dialog) : $('.batch-content', dialog);
    (textarea && !textarea.disabled ? textarea : $('.batch-editor-cancel', dialog)).focus();
  }
  function setEditorBusy(form, busy) {
    form._busy = busy;
    form.setAttribute('aria-busy', String(busy));
    $$('input, textarea, button', form).forEach(input => { input.disabled = busy; });
    if (!busy) updateEditor(form);
  }
  function showRequestError(event, message) {
    const form = event.detail.requestConfig?.elt || event.detail.elt;
    if (form?.id === 'batch-editor-form' && form.isConnected) {
      const error = $('#batch-editor-error', form);
      error.textContent = message;
      error.hidden = false;
      error.scrollIntoView({ block: 'center' });
    } else toast(message, true);
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
    initializeEditor();
    if (form && !readyForms.has(form)) {
      readyForms.add(form);
      const textarea = $('textarea', form);
      textarea.value = storage('getItem', draftKey(form)) || '';
      resize(textarea);
      if (form.dataset.mode === 'chat') {
        let saved = {};
        try { saved = JSON.parse(storage('getItem', searchKey(form)) || '{}') || {}; } catch (_) { /* Ignore invalid saved settings. */ }
        $('[name="web_search"]', form).checked = form.hasAttribute('data-editor-web-search') ? form.dataset.editorWebSearch === 'true' : saved.enabled === true;
        $('.attachment-input', form).value = '';
        saveSearch(form);
        renderAttachments(form);
      }
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
  document.addEventListener('change', event => {
    const form = event.target.closest('form.composer');
    if (!form) return;
    if (event.target.matches('[name="web_search"]')) saveSearch(form);
    if (event.target.matches('.attachment-input')) {
      const entries = attachmentState(form);
      for (const file of event.target.files) {
        const preview = ['image/png', 'image/jpeg', 'image/webp'].includes(file.type);
        entries.push({file, url: preview ? URL.createObjectURL(file) : null});
      }
      event.target.value = '';
      renderAttachments(form);
    }
  });
  document.addEventListener('keydown', event => {
    if ($('#batch-editor[open]')) {
      if ((event.metaKey || event.ctrlKey) && ['k', 'enter'].includes(event.key.toLowerCase())) event.preventDefault();
      return;
    }
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
      if (!$('button[type="submit"]', form).disabled && hasContent(form)) form.requestSubmit();
    }
  });
  document.addEventListener('click', event => {
    const target = event.target;
    if (target.closest('.batch-editor-cancel')) { closeEditor($('#batch-editor')); return; }
    if (!target.closest('.params-toggle')) $$('.params-toggle[open]').forEach(details => { details.open = false; });
    if (target.closest('.menu-btn')) drawer(!$('.layout').classList.contains('nav-open'));
    if (target.closest('.scrim')) drawer(false);
    if (target.closest('.jump-latest')) { followLatest = true; scrollBottom(); syncMessages(); }
    const inputToggle = target.closest('.input-tools-toggle');
    if (inputToggle) {
      const panel = $('.composer-extras', inputToggle.closest('form'));
      panel.hidden = !panel.hidden;
      inputToggle.setAttribute('aria-expanded', String(!panel.hidden));
    }
    const removeAttachment = target.closest('.remove-attachment');
    if (removeAttachment) {
      const form = removeAttachment.closest('form');
      const [removed] = attachmentState(form).splice(Number(removeAttachment.dataset.attachmentIndex), 1);
      if (removed?.url) URL.revokeObjectURL(removed.url);
      renderAttachments(form);
    }
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
    const copyPrompt = target.closest('#copy-prompt-btn');
    if (copyPrompt) copy($('#prompt-messages').textContent, copyPrompt);
    const copyAll = target.closest('#copy-all-btn');
    if (copyAll) copy($$('#message-list .message').map(el => `${$('.role-label', el)?.textContent || ''}\n${$('.content', el)?.textContent || ''}`).join('\n\n'), copyAll);
  });
  document.body.addEventListener('htmx:configRequest', event => {
    const form = event.detail.elt;
    if (form.id === 'batch-editor-form') {
      updateEditor(form);
      event.detail.parameters.draft = JSON.stringify(editorDraft(form));
    }
    const composer = $('form.composer');
    if (form.matches('form.composer, .request-response, .regenerate-response, .retry-generation')) {
      event.detail.parameters.web_search = $('[name="web_search"]', composer || document)?.checked ? 'true' : 'false';
    }
    if (!form.matches('form.composer') || form.dataset.mode !== 'chat') return;
    const entries = attachmentState(form);
    // An empty native file input becomes files='' in FormData; omit it entirely.
    delete event.detail.parameters.files;
    if (entries.length) event.detail.parameters.files = entries.map(entry => entry.file);
    form._submittedFiles = [...entries];
  });
  document.body.addEventListener('htmx:beforeRequest', event => {
    const form = event.detail.elt;
    const title = $('.title-input', form);
    if (title) form._submittedTitle = title.value;
    const sendsMessage = form.matches('form.composer');
    const editsMessages = form.id === 'batch-editor-form';
    const mutatesMessage = form.matches('[data-message-mutation]');
    if (sendsMessage || mutatesMessage || editsMessages) {
      if (activeMessageRequest || $('#message-list [data-generation-status="queued"], #message-list [data-generation-status="running"]')) {
        event.preventDefault();
        if (editsMessages) toast('The source chat is busy. Your edits remain open; wait until it is idle before saving.', true);
        return;
      }
      if (sendsMessage && !hasContent(form)) { event.preventDefault(); return; }
      activeMessageRequest = event.detail.xhr;
      if (sendsMessage) {
        form._submittedDraft = $('textarea', form).value;
        followLatest = true;
      }
      syncMessages();
      if (editsMessages) {
        $('#batch-editor-error', form).hidden = true;
        setEditorBusy(form, true);
      }
    }
  });
  document.body.addEventListener('htmx:afterRequest', event => {
    const form = event.detail.requestConfig?.elt || event.detail.elt;
    // htmx retargets afterRequest when a message action replaces its own button.
    if (activeMessageRequest === event.detail.xhr) activeMessageRequest = null;
    if (form.id === 'batch-editor-form' && form.isConnected) setEditorBusy(form, false);
    const title = $('.title-input', form);
    const savedTitle = $('.session-item.active .session-title');
    if (event.detail.successful && title && savedTitle && title.value === form._submittedTitle) {
      title.value = savedTitle.textContent;
    }
    if (event.detail.successful && form.matches('form.composer')) {
      const currentForm = $$('form.composer').find(element => element.dataset.sessionId === form.dataset.sessionId);
      const textarea = currentForm ? $('textarea', currentForm) : null;
      if (storage('getItem', draftKey(form)) === form._submittedDraft) storage('removeItem', draftKey(form));
      if (textarea && textarea.value === form._submittedDraft) {
        textarea.value = ''; resize(textarea);
      }
      if (form.dataset.mode === 'chat') {
        const submitted = new Set(form._submittedFiles || []);
        const retained = attachmentState(form).filter(entry => {
          if (!submitted.has(entry)) return true;
          if (entry.url) URL.revokeObjectURL(entry.url);
          return false;
        });
        attachmentDrafts.set(form.dataset.sessionId, retained);
        if (currentForm) renderAttachments(currentForm);
      }
      if (form.isConnected) textarea?.focus({preventScroll: true});
    }
    syncMessages();
  });
  ['htmx:afterSwap', 'htmx:afterSettle', 'htmx:oobAfterSwap', 'htmx:historyRestore'].forEach(name => document.body.addEventListener(name, () => requestAnimationFrame(initialize)));
  document.body.addEventListener('htmx:responseError', event => {
    let message = 'The request failed. Please try again.';
    try {
      const detail = JSON.parse(event.detail.xhr.responseText).detail;
      message = typeof detail === 'string' ? detail : detail.map(error => `${error.loc.at(-1)}: ${error.msg}`).join(' · ');
    } catch (_) { /* Non-JSON errors use the safe fallback. */ }
    showRequestError(event, message);
  });
  ['htmx:sendError', 'htmx:timeout'].forEach(name => document.body.addEventListener(name, event =>
    showRequestError(event, 'Connection lost. Please try again.')));
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
    const textarea = $('.composer textarea');
    if (textarea) resize(textarea);
  });
  document.addEventListener('scroll', event => {
    if (event.target.matches?.('#folder-list, .sidebar')) closeFolderMenus();
  }, true);
  initialize();
  dispatchEvent(new Event('resize'));
})();
