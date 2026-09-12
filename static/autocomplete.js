function hideProximityError() {
    var el = document.getElementById('proximity-error-toast');
    if (el) { el.style.animation = 'none'; el.style.opacity = '0'; el.style.display = 'none'; }
}

function showFields() {
    var mode = document.querySelector('input[name="mode"]:checked').value;
    document.getElementById('fields-freetext').style.display = (mode === 'freetext') ? 'block' : 'none';
    document.getElementById('fields-phrase').style.display = (mode === 'phrase') ? 'block' : 'none';
    document.getElementById('fields-proximity').style.display = (mode === 'proximity') ? 'block' : 'none';
    if (mode !== 'proximity') hideProximityError();
}

window.addEventListener('DOMContentLoaded', function () {
    var toast = document.getElementById('proximity-error-toast');
    if (toast) {
        toast.addEventListener('animationend', function () {
            toast.style.display = 'none';
        });
    }
    setupAutocomplete('ac-query');
    setupAutocomplete('ac-phrase');
    setupAutocomplete('ac-prox1');
    setupAutocomplete('ac-prox2');
});

// Generic autocomplete wiring: takes the id prefix of an
// { <id>-input, <id>-dropdown } pair and hooks up fetch + keyboard nav.
function setupAutocomplete(idPrefix) {
    var input = document.getElementById(idPrefix + '-input');
    var dropdown = document.getElementById(idPrefix + '-dropdown');
    if (!input || !dropdown) return;

    var activeIndex = -1;
    var debounceTimer = null;

    function closeDropdown() {
        dropdown.style.display = 'none';
        dropdown.innerHTML = '';
        activeIndex = -1;
    }

    function renderSuggestions(items) {
        dropdown.innerHTML = '';
        if (!items.length) { closeDropdown(); return; }
        items.forEach(function (term) {
            var div = document.createElement('div');
            div.textContent = term;
            div.addEventListener('mousedown', function (e) {
                e.preventDefault();
                input.value = term;
                closeDropdown();
            });
            dropdown.appendChild(div);
        });
        dropdown.style.display = 'block';
    }

    input.addEventListener('input', function () {
        var q = input.value.trim();
        clearTimeout(debounceTimer);
        if (!q) { closeDropdown(); return; }
        debounceTimer = setTimeout(function () {
            fetch('/autocomplete?q=' + encodeURIComponent(q))
                .then(function (res) { return res.json(); })
                .then(function (data) { renderSuggestions(data.suggestions || []); })
                .catch(function () { closeDropdown(); });
        }, 120);
    });

    input.addEventListener('keydown', function (e) {
        var items = dropdown.querySelectorAll('div');
        if (!items.length || dropdown.style.display === 'none') return;

        if (e.key === 'ArrowDown') {
            e.preventDefault();
            activeIndex = Math.min(activeIndex + 1, items.length - 1);
            updateActive(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            activeIndex = Math.max(activeIndex - 1, 0);
            updateActive(items);
        } else if (e.key === 'Enter') {
            if (activeIndex >= 0) {
                e.preventDefault();
                input.value = items[activeIndex].textContent;
                closeDropdown();
            }
        } else if (e.key === 'Escape') {
            closeDropdown();
        }
    });

    function updateActive(items) {
        items.forEach(function (el, i) {
            el.classList.toggle('ac-active', i === activeIndex);
        });
    }

    document.addEventListener('click', function (e) {
        if (e.target !== input) closeDropdown();
    });
}