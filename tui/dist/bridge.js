import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..', '..');
const SERVER = path.join(__dirname, '..', 'server.py');
let serverProcess = null;
let pending = new Map();
function ensureServer(modelId) {
    if (serverProcess)
        return;
    const py = process.platform === 'win32' ? 'python' : 'python3';
    const args = [SERVER];
    if (modelId)
        args.push('--model-id', modelId);
    const proc = spawn('python', args, {
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
    });
    serverProcess = proc;
    if (proc.stderr) {
        proc.stderr.on('data', d => console.error('[server]', d.toString().trim()));
    }
    proc.on('error', e => console.error('[server] spawn error', e));
    proc.on('close', () => { serverProcess = null; pending.clear(); });
    let buffer = '';
    if (proc.stdout) {
        proc.stdout.on('data', d => {
            buffer += d.toString();
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.trim())
                    continue;
                try {
                    const res = JSON.parse(line);
                    const cb = pending.get(res.ok ? 'ok' : 'err');
                }
                catch { }
            }
        });
    }
}
let resolveNext = null;
function sendRequest(text, effort, modelId) {
    if (!serverProcess)
        ensureServer(modelId);
    return new Promise(resolve => {
        resolveNext = resolve;
        const proc = serverProcess;
        if (!proc)
            return resolve({ ok: false, error: 'server failed to start' });
        const req = JSON.stringify({ text, effort, model_id: modelId }) + '\n';
        if (proc.stdin)
            proc.stdin.write(req);
    });
}
export function humanizeViaPython(text, effort, modelId) {
    return sendRequest(text, effort, modelId);
}
function attachStdout() {
    const proc = serverProcess;
    if (!proc || !proc.stdout)
        return;
    proc.stdout.on('data', d => {
        const lines = d.toString().split('\n').filter(Boolean);
        for (const line of lines) {
            try {
                const res = JSON.parse(line);
                if (resolveNext) {
                    resolveNext(res);
                    resolveNext = null;
                }
            }
            catch { }
        }
    });
}
attachStdout();
export function setModelId(modelId) {
    if (serverProcess) {
        serverProcess.kill();
        serverProcess = null;
    }
}
