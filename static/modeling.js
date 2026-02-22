document.addEventListener('DOMContentLoaded', () => {
    const docsContainer = document.getElementById('docs-container');
    const addDocBtn = document.getElementById('add-doc');
    const runBtn = document.getElementById('btn-run');
    const methodSelect = document.getElementById('method-select');
    const resultsSection = document.getElementById('results-section');
    const tableHeader = document.getElementById('table-header');
    const tableBody = document.getElementById('table-body');
    const methodLabel = document.getElementById('current-method-label');

    addDocBtn.addEventListener('click', () => {
        const textarea = document.createElement('textarea');
        textarea.placeholder = `${docsContainer.children.length + 1}-hujjat matni...`;
        docsContainer.appendChild(textarea);
    });

    runBtn.addEventListener('click', async () => {
        const textareas = docsContainer.querySelectorAll('textarea');
        const texts = Array.from(textareas)
            .map(t => t.value.trim())
            .filter(t => t.length > 0);

        if (texts.length === 0) {
            alert("Iltimos, kamida bitta hujjat kiriting.");
            return;
        }

        const method = methodSelect.value;
        runBtn.disabled = true;
        runBtn.textContent = 'Yuklanmoqda...';

        try {
            const response = await fetch('/analyze_modeling', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ texts, method })
            });

            if (!response.ok) throw new Error('Modeling failed');

            const result = await response.json();
            renderTable(result);
        } catch (err) {
            console.error(err);
            alert("Xatolik yuz berdi: " + err.message);
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = 'Modellashtirish';
        }
    });

    function renderTable(result) {
        resultsSection.style.display = 'block';
        methodLabel.textContent = result.method.toUpperCase();

        // Header
        tableHeader.innerHTML = '<th class="doc-col">Hujjat</th>' + 
            result.features.map(f => `<th>${f}</th>`).join('');

        // Body
        tableBody.innerHTML = result.data.map(row => {
            const cells = row.vector.map(v => {
                const opacity = v > 0 ? Math.min(0.1 + v, 0.8) : 0;
                const style = v > 0 ? `style="background: rgba(108, 92, 231, ${opacity})"` : '';
                return `<td class="val-cell" ${style}>${v}</td>`;
            }).join('');
            return `<tr><td class="doc-col">${row.text}</td>${cells}</tr>`;
        }).join('');
    }
});
