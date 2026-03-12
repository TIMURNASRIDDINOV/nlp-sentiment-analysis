document.addEventListener('DOMContentLoaded', () => {
    // ==================================================================
    //  Global DOM refs
    // ==================================================================
    const tabBtns      = document.querySelectorAll('.tab-btn[data-tab]');
    const toolViews    = document.querySelectorAll('.tool-view');
    const themeToggle  = document.getElementById('theme-toggle');
    const modelInfo    = document.getElementById('model-info');

    // -- Sentiment refs -------------------------------------------------
    const sentimentInput   = document.getElementById('sentiment-input');
    const sentimentBtn     = document.getElementById('btn-sentiment');
    const sentimentLoader  = document.getElementById('sentiment-loader');
    const sentimentResults = document.getElementById('sentiment-results');
    const sentimentElapsed = document.getElementById('sentiment-elapsed');
    const topKSelect       = document.getElementById('top-k');

    // -- Modeling refs --------------------------------------------------
    const docsContainer   = document.getElementById('docs-container');
    const addDocBtn       = document.getElementById('btn-add-doc');
    const modelingBtn     = document.getElementById('btn-modeling');
    const methodSelect    = document.getElementById('method-select');
    const chkLowercase    = document.getElementById('chk-lowercase');
    const chkStopwords    = document.getElementById('chk-stopwords');
    const modelingLoader  = document.getElementById('modeling-loader');
    const modelingResults = document.getElementById('modeling-results');

    let freqChart    = null;
    let weightsChart = null;

    // ==================================================================
    //  Tab switching
    // ==================================================================
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            toolViews.forEach(v => {
                v.classList.toggle('active', v.id === `${tab}-view`);
            });
        });
    });

    // ==================================================================
    //  Theme toggle
    // ==================================================================
    function applyTheme(dark) {
        document.body.setAttribute('data-theme', dark ? 'dark' : 'light');
        themeToggle.textContent = dark ? 'Kunduzgi rejim' : 'Tungi rejim';
        localStorage.setItem('theme', dark ? 'dark' : 'light');
    }
    applyTheme((localStorage.getItem('theme') || 'dark') === 'dark');
    themeToggle.addEventListener('click', () => {
        applyTheme(document.body.getAttribute('data-theme') !== 'dark');
    });

    // ==================================================================
    //  Model info
    // ==================================================================
    (async () => {
        try {
            const res = await fetch('/models');
            if (!res.ok) throw 0;
            const d = await res.json();
            modelInfo.textContent = `Model: ${d.current_model} | ${d.device === -1 ? 'CPU' : 'GPU ' + d.device}`;
        } catch { modelInfo.textContent = 'Tizim holati: Tayyor'; }
    })();

    // ==================================================================
    //  Sentiment engine (unchanged logic)
    // ==================================================================
    sentimentBtn.addEventListener('click', async () => {
        const text = sentimentInput.value.trim();
        if (!text) return alert('Iltimos, matn kiriting');
        const texts = text.split('\n').map(l => l.trim()).filter(Boolean);
        const topK  = parseInt(topKSelect.value) || 1;

        sentimentBtn.disabled = true;
        sentimentLoader.classList.remove('hidden');
        sentimentResults.classList.add('hidden');
        sentimentElapsed.classList.add('hidden');

        try {
            const res  = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type':'application/json'},
                body: JSON.stringify({texts, top_k: topK}),
            });
            const data = await res.json();
            renderSentiment(data);
        } catch { alert('Sentiment tahlilida xatolik'); }
        finally {
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
                const cls = s.label.toLowerCase().includes('pos') ? 'positive' :
                            s.label.toLowerCase().includes('neg') ? 'negative' : 'neutral';
                const badge = document.createElement('span');
                badge.className = `badge ${cls}`;
                badge.textContent = `${s.label} (${(s.score*100).toFixed(1)}%)`;
                const track = document.createElement('div');
                track.className = 'score-track';
                const fill = document.createElement('div');
                fill.className = `score-fill ${cls}`;
                fill.style.width = '0%';
                track.appendChild(fill);
                card.appendChild(badge);
                card.appendChild(track);
                setTimeout(() => fill.style.width = `${s.score*100}%`, 50);
            });
            sentimentResults.appendChild(card);
        });
        sentimentElapsed.textContent = `Vaqt: ${data.elapsed_ms} ms`;
        sentimentElapsed.classList.remove('hidden');
        sentimentResults.classList.remove('hidden');
    }

    // ==================================================================
    //  Modeling Lab
    // ==================================================================
    addDocBtn.addEventListener('click', () => {
        const ta = document.createElement('textarea');
        ta.placeholder = `${docsContainer.children.length + 1}-hujjat...`;
        ta.style.marginTop = '0.5rem';
        docsContainer.appendChild(ta);
    });

    modelingBtn.addEventListener('click', async () => {
        const texts = Array.from(docsContainer.querySelectorAll('textarea'))
            .map(t => t.value.trim()).filter(Boolean);
        if (texts.length < 2) return alert('Kamida 2 ta hujjat kiriting');

        const method         = methodSelect.value;
        const lowercase      = chkLowercase.checked;
        const remove_stopwords = chkStopwords.checked;

        modelingBtn.disabled = true;
        modelingLoader.classList.remove('hidden');
        modelingResults.classList.add('hidden');

        try {
            if (method === 'compare') {
                const [bowRes, tfidfRes] = await Promise.all([
                    callModeling(texts, 'bow', lowercase, remove_stopwords),
                    callModeling(texts, 'tfidf', lowercase, remove_stopwords),
                ]);
                renderCompare(bowRes, tfidfRes);
            } else {
                const result = await callModeling(texts, method, lowercase, remove_stopwords);
                renderSingleMode(result);
            }
        } catch (e) {
            console.error(e);
            alert('Modellashtirishda xatolik: ' + e.message);
        } finally {
            modelingBtn.disabled = false;
            modelingLoader.classList.add('hidden');
        }
    });

    async function callModeling(texts, method, lowercase, remove_stopwords) {
        const res = await fetch('/analyze_modeling', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({texts, method, lowercase, remove_stopwords}),
        });
        if (!res.ok) throw new Error('API xatolik');
        return res.json();
    }

    // ------------------------------------------------------------------
    //  Single mode rendering (BoW or TF-IDF)
    // ------------------------------------------------------------------
    function renderSingleMode(r) {
        const isTfidf = r.method === 'tfidf';

        renderTokens(r);
        renderVocab(r);

        if (isTfidf) {
            renderStep3Tfidf(r);
            show('step4-card');  renderStep4(r);
            show('step5-card');  renderStep5(r);
            setNum('step6-num', '06'); setNum('step7-num', '07');
        } else {
            renderStep3Bow(r);
            hide('step4-card');
            hide('step5-card');
            setNum('step6-num', '04'); setNum('step7-num', '05');
        }

        renderStep6(r);
        renderStep7(r);
        hide('compare-card');
        renderCharts(r);

        modelingResults.classList.remove('hidden');
    }

    // ------------------------------------------------------------------
    //  Compare mode rendering
    // ------------------------------------------------------------------
    function renderCompare(bow, tfidf) {
        renderTokens(tfidf);
        renderVocab(tfidf);

        renderStep3Tfidf(tfidf);
        show('step4-card');  renderStep4(tfidf);
        show('step5-card');  renderStep5(tfidf);

        document.getElementById('step6-method').textContent = 'TF-IDF';
        setNum('step6-num', '06'); setNum('step7-num', '07');
        renderStep6(tfidf);
        renderStep7(tfidf);

        show('compare-card');
        fillTable('cmp-bow-header', 'cmp-bow-body', bow.features, bow.data, true);
        fillTable('cmp-tfidf-header', 'cmp-tfidf-body', tfidf.features, tfidf.data, false);
        fillSimTable('cmp-bow-sim-h', 'cmp-bow-sim-b', bow.similarity_matrix, bow.documents);
        fillSimTable('cmp-tfidf-sim-h', 'cmp-tfidf-sim-b', tfidf.similarity_matrix, tfidf.documents);

        renderCharts(tfidf);
        modelingResults.classList.remove('hidden');
    }

    // ------------------------------------------------------------------
    //  Step renderers
    // ------------------------------------------------------------------
    function renderTokens(r) {
        const el = document.getElementById('step-tokens');
        el.innerHTML = r.steps.tokenized_docs.map((tokens, i) =>
            `<div class="token-row">
                <span class="token-label">Doc ${i+1}:</span>
                <span class="token-list">${tokens.map(t => `<span class="token">${t}</span>`).join('')}</span>
            </div>`
        ).join('');
    }

    function renderVocab(r) {
        const el = document.getElementById('step-vocab');
        el.innerHTML = `<div class="vocab-cloud">${r.steps.vocabulary.map(w =>
            `<span class="vocab-word">${w}</span>`).join('')}</div>
            <p class="step-meta">Jami: <strong>${r.steps.vocabulary.length}</strong> ta noyob so'z</p>`;
    }

    function renderStep3Bow(r) {
        document.getElementById('step3-title').textContent = 'Bag of Words (so\'z soni)';
        document.getElementById('step3-formula').innerHTML = '<code>BoW(t, d) = t so\'zi d hujjatda necha marta uchrashi</code>';
        fillTable('step3-header', 'step3-body', r.features, r.data, true);
    }

    function renderStep3Tfidf(r) {
        document.getElementById('step3-title').textContent = 'Term Frequency (TF)';
        document.getElementById('step3-formula').innerHTML = '<code>TF(t, d) = count(t, d) / |d|</code>&nbsp;&nbsp;(|d| = hujjatdagi so\'zlar soni)';
        const tfData = r.steps.tf_matrix.map((vec, i) => ({text: r.documents[i], vector: vec}));
        fillTable('step3-header', 'step3-body', r.features, tfData, false);
    }

    function renderStep4(r) {
        const body = document.getElementById('step4-body');
        body.innerHTML = r.features.map(f =>
            `<tr><td class="doc-col">${f}</td><td>${r.steps.df[f]}</td><td>${r.steps.idf[f]}</td></tr>`
        ).join('');
    }

    function renderStep5(r) {
        const rawData = r.steps.tfidf_raw.map((vec, i) => ({text: r.documents[i], vector: vec}));
        fillTable('step5-header', 'step5-body', r.features, rawData, false);
    }

    function renderStep6(r) {
        const methodName = r.method === 'tfidf' ? 'TF-IDF (L2 normalizatsiya)' : 'Bag of Words';
        document.getElementById('step6-method').textContent = methodName;
        const isInt = r.method === 'bow';
        if (isInt) {
            document.getElementById('step6-formula').innerHTML = '<code>Yakuniy matritsa: har bir katak = so\'z soni</code>';
        } else {
            document.getElementById('step6-formula').innerHTML = '<code>L2 normalizatsiya: v&#x0302; = v / &Vert;v&Vert;</code>';
        }
        fillTable('step6-header', 'step6-body', r.features, r.data, isInt);
    }

    function renderStep7(r) {
        fillSimTable('sim-header', 'sim-body', r.similarity_matrix, r.documents);
    }

    // ------------------------------------------------------------------
    //  Table builders
    // ------------------------------------------------------------------
    function fillTable(headerId, bodyId, features, data, isInt) {
        document.getElementById(headerId).innerHTML =
            '<th class="doc-col">Hujjat</th>' + features.map(f => `<th>${f}</th>`).join('');
        document.getElementById(bodyId).innerHTML = data.map(row => {
            const cells = row.vector.map(v => {
                const val = isInt ? v : v;
                const maxV = isInt ? Math.max(...row.vector, 1) : 1;
                const ratio = isInt ? v / maxV : v;
                const op = ratio > 0 ? Math.min(0.1 + ratio * 0.6, 0.7) : 0;
                const style = ratio > 0
                    ? `style="background:rgba(99,102,241,${op});color:${op>0.45?'#fff':'inherit'}"`
                    : '';
                return `<td ${style}>${isInt ? val : val.toFixed ? val.toFixed(4) : val}</td>`;
            }).join('');
            return `<tr><td class="doc-col">${truncate(row.text, 50)}</td>${cells}</tr>`;
        }).join('');
    }

    function fillSimTable(headerId, bodyId, matrix, docs) {
        document.getElementById(headerId).innerHTML =
            '<th></th>' + docs.map((_, i) => `<th>Doc ${i+1}</th>`).join('');
        document.getElementById(bodyId).innerHTML = matrix.map((row, i) =>
            `<tr><td class="doc-col"><strong>Doc ${i+1}</strong></td>${row.map(v => {
                const hue = v >= 0.7 ? '145' : v >= 0.3 ? '45' : '0';
                const sat = v > 0 ? '70%' : '0%';
                const op = Math.min(0.15 + v * 0.55, 0.7);
                return `<td style="background:hsla(${hue},${sat},50%,${op});font-weight:600">${v.toFixed(4)}</td>`;
            }).join('')}</tr>`
        ).join('');
    }

    // ------------------------------------------------------------------
    //  Charts (Chart.js)
    // ------------------------------------------------------------------
    function renderCharts(r) {
        if (freqChart)    { freqChart.destroy();    freqChart = null; }
        if (weightsChart) { weightsChart.destroy(); weightsChart = null; }

        const isDark = document.body.getAttribute('data-theme') === 'dark';
        const gridColor = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
        const textColor = isDark ? '#a1a1aa' : '#64748b';

        Chart.defaults.color = textColor;

        const totalCounts = r.features.map((_, j) =>
            r.steps.bow_matrix.reduce((sum, row) => sum + row[j], 0)
        );

        freqChart = new Chart(document.getElementById('chart-freq'), {
            type: 'bar',
            data: {
                labels: r.features,
                datasets: [{
                    label: 'Umumiy chastota (barcha hujjatlarda)',
                    data: totalCounts,
                    backgroundColor: 'rgba(99,102,241,0.7)',
                    borderColor: 'rgba(99,102,241,1)',
                    borderWidth: 1,
                    borderRadius: 4,
                }],
            },
            options: {
                indexAxis: r.features.length > 8 ? 'y' : 'x',
                responsive: true,
                plugins: {
                    legend: { display: false },
                    title: { display: true, text: "So'zlar chastotasi (Bag of Words)", font: { size: 14 } },
                },
                scales: {
                    x: { grid: { color: gridColor } },
                    y: { grid: { color: gridColor } },
                },
            },
        });

        const colors = [
            'rgba(99,102,241,0.7)', 'rgba(244,63,94,0.7)', 'rgba(16,185,129,0.7)',
            'rgba(245,158,11,0.7)', 'rgba(139,92,246,0.7)', 'rgba(6,182,212,0.7)',
        ];
        const borders = [
            'rgba(99,102,241,1)', 'rgba(244,63,94,1)', 'rgba(16,185,129,1)',
            'rgba(245,158,11,1)', 'rgba(139,92,246,1)', 'rgba(6,182,212,1)',
        ];

        weightsChart = new Chart(document.getElementById('chart-weights'), {
            type: 'bar',
            data: {
                labels: r.features,
                datasets: r.data.map((row, i) => ({
                    label: `Doc ${i+1}`,
                    data: row.vector.map(v => typeof v === 'number' ? +v.toFixed(4) : v),
                    backgroundColor: colors[i % colors.length],
                    borderColor: borders[i % borders.length],
                    borderWidth: 1,
                    borderRadius: 3,
                })),
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `Hujjat vektorlari (${r.method.toUpperCase()})`,
                        font: { size: 14 },
                    },
                },
                scales: {
                    x: { grid: { color: gridColor } },
                    y: { grid: { color: gridColor }, beginAtZero: true },
                },
            },
        });
    }

    // ------------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------------
    function show(id) { document.getElementById(id).classList.remove('hidden'); }
    function hide(id) { document.getElementById(id).classList.add('hidden'); }
    function setNum(id, val) { document.getElementById(id).textContent = val; }
    function truncate(s, n) { return s.length > n ? s.slice(0, n) + '...' : s; }
});
