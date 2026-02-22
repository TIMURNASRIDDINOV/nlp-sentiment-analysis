document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Refs: Global --------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn[data-tab]');
    const toolViews = document.querySelectorAll('.tool-view');
    const themeToggle = document.getElementById('theme-toggle');
    const modelInfo = document.getElementById('model-info');

    // ---- DOM Refs: Sentiment ----------------------------------------------
    const sentimentInput = document.getElementById('sentiment-input');
    const sentimentBtn = document.getElementById('btn-sentiment');
    const sentimentLoader = document.getElementById('sentiment-loader');
    const sentimentResults = document.getElementById('sentiment-results');
    const sentimentElapsed = document.getElementById('sentiment-elapsed');
    const topKSelect = document.getElementById('top-k');

    // ---- DOM Refs: Modeling -----------------------------------------------
    const docsContainer = document.getElementById('docs-container');
    const addDocBtn = document.getElementById('btn-add-doc');
    const modelingBtn = document.getElementById('btn-modeling');
    const methodSelect = document.getElementById('method-select');
    const modelingLoader = document.getElementById('modeling-loader');
    const modelingResultsContainer = document.getElementById('modeling-results-container');
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');
    const methodLabel = document.getElementById('current-method-lbl');

    // ---- Tab Switching -----------------------------------------------------
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            toolViews.forEach(v => {
                v.classList.remove('active');
                if (v.id === `${tab}-view`) v.classList.add('active');
            });
        });
    });

    // ---- Theme Toggle ------------------------------------------------------
    function applyTheme(dark) {
        document.body.setAttribute('data-theme', dark ? 'dark' : 'light');
        themeToggle.textContent = dark ? '☀️ Kunduzgi rejim' : '🌙 Tungi rejim';
        localStorage.setItem('theme', dark ? 'dark' : 'light');
    }

    const savedTheme = localStorage.getItem('theme') || 'dark';
    applyTheme(savedTheme === 'dark');

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.getAttribute('data-theme') === 'dark';
        applyTheme(!isDark);
    });

    // ---- Fetch Model Info -------------------------------------------------
    async function loadModelInfo() {
        try {
            const res = await fetch("/models");
            if (!res.ok) throw new Error();
            const data = await res.json();
            modelInfo.textContent = `Model: ${data.current_model} | Device: ${data.device === -1 ? "CPU" : "GPU " + data.device}`;
        } catch {
            modelInfo.textContent = "Tizim holati: Tayyor";
        }
    }
    loadModelInfo();

    // ---- Sentiment Engine Logic -------------------------------------------
    sentimentBtn.addEventListener('click', async () => {
        const text = sentimentInput.value.trim();
        if (!text) return alert("Iltimos, matn kiriting");

        const texts = text.split('\n').map(l => l.trim()).filter(Boolean);
        const topK = parseInt(topKSelect.value) || 1;

        sentimentBtn.disabled = true;
        sentimentLoader.classList.remove('hidden');
        sentimentResults.classList.add('hidden');
        sentimentElapsed.classList.add('hidden');

        try {
            const res = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts, top_k: topK })
            });
            const data = await res.json();
            renderSentiment(data);
        } catch (err) {
            alert("Sentiment tahlilida xatolik");
        } finally {
            sentimentBtn.disabled = false;
            sentimentLoader.classList.add('hidden');
        }
    });

    function renderSentiment(data) {
        sentimentResults.innerHTML = '';
        data.results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'sentiment-card';
            
            const txt = document.createElement('div');
            txt.style.fontWeight = '500';
            txt.textContent = item.text;
            card.appendChild(txt);

            item.sentiments.forEach(s => {
                const labelClass = s.label.toLowerCase().includes('pos') ? 'positive' : 
                                 s.label.toLowerCase().includes('neg') ? 'negative' : 'neutral';
                
                const badge = document.createElement('span');
                badge.className = `badge ${labelClass}`;
                badge.textContent = `${s.label} (${(s.score * 100).toFixed(1)}%)`;
                
                const track = document.createElement('div');
                track.className = 'score-track';
                const fill = document.createElement('div');
                fill.className = `score-fill ${labelClass}`;
                fill.style.width = '0%';
                track.appendChild(fill);
                
                card.appendChild(badge);
                card.appendChild(track);
                setTimeout(() => fill.style.width = `${s.score * 100}%`, 50);
            });
            sentimentResults.appendChild(card);
        });
        
        sentimentElapsed.textContent = `Vaqt: ${data.elapsed_ms} ms`;
        sentimentElapsed.classList.remove('hidden');
        sentimentResults.classList.remove('hidden');
    }

    // ---- Modeling Lab Logic -----------------------------------------------
    addDocBtn.addEventListener('click', () => {
        const textarea = document.createElement('textarea');
        textarea.placeholder = `${docsContainer.children.length + 1}-hujjat...`;
        textarea.style.marginTop = '0.5rem';
        docsContainer.appendChild(textarea);
    });

    modelingBtn.addEventListener('click', async () => {
        const textareas = docsContainer.querySelectorAll('textarea');
        const texts = Array.from(textareas).map(t => t.value.trim()).filter(Boolean);
        
        if (texts.length === 0) return alert("Hujjat matnini kiriting");

        modelingBtn.disabled = true;
        modelingLoader.classList.remove('hidden');
        modelingResultsContainer.classList.add('hidden');

        try {
            const res = await fetch('/analyze_modeling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts, method: methodSelect.value })
            });
            const data = await res.json();
            renderModeling(data);
        } catch (err) {
            alert("Modellashtirishda xatolik");
        } finally {
            modelingBtn.disabled = false;
            modelingLoader.classList.add('hidden');
        }
    });

    function renderModeling(result) {
        methodLabel.textContent = result.method.toUpperCase();
        tableHeader.innerHTML = '<th class="doc-col">Hujjat</th>' + 
            result.features.map(f => `<th>${f}</th>`).join('');

        tableBody.innerHTML = result.data.map(row => {
            const cells = row.vector.map(v => {
                const opacity = v > 0 ? Math.min(0.1 + v, 0.7) : 0;
                const style = v > 0 ? `style="background: rgba(129, 140, 248, ${opacity}); color: ${v > 0.4 ? 'white' : 'inherit'}"` : '';
                return `<td ${style}>${v}</td>`;
            }).join('');
            return `<tr><td class="doc-col">${row.text}</td>${cells}</tr>`;
        }).join('');

        modelingResultsContainer.classList.remove('hidden');
    }
});
