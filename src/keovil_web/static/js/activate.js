async function attemptHandshake() {
    const btn = document.getElementById('btn');
    const feedback = document.getElementById('feedback');
    const key = document.getElementById('masterKey').value;

    btn.innerText = "SYNCHRONIZING...";
    btn.disabled = true;
    feedback.classList.add('hidden');

    try {
        const response = await fetch('/api/bootstrap', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ master_key: key })
        });

        const data = await response.json();

        if (response.ok) {
            btn.innerText = "AUTHORIZED";
            btn.classList.replace('bg-white', 'bg-green-500');
            setTimeout(() => window.location.href = "/", 1000);
        } else {
            throw new Error(data.msg || "Authorization Denied");
        }
    } catch (err) {
        feedback.innerText = `ERROR: ${err.message}`;
        feedback.classList.remove('hidden');
        btn.innerText = "INITIALIZE HANDSHAKE";
        btn.disabled = false;
    }
}
