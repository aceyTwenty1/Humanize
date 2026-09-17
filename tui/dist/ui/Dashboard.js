import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { useState, useMemo } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import chalk from 'chalk';
import { SlashOverlay } from './SlashOverlay.js';
import { humanizeViaPython } from '../bridge.js';
function ScoreBar({ score, target }) {
    const met = score >= target;
    const width = 24;
    const fill = Math.round(Math.min(1, Math.max(0, score)) * width);
    const bar = '█'.repeat(fill) + '░'.repeat(width - fill);
    const color = met ? chalk.hex('#5FBF7A') : score > 0.4 ? chalk.hex('#E0A83C') : chalk.red;
    return (_jsxs(Text, { children: [color(`[${bar}]`), " ", chalk.whiteBright(`${score.toFixed(3)} / ${target.toFixed(2)}`), " ", met ? chalk.green('✓ PASS') : chalk.yellow('✗ retry')] }));
}
export function Dashboard({ config }) {
    const { exit } = useApp();
    const [target] = useState(0.85);
    const [input, setInput] = useState('');
    const [busy, setBusy] = useState(false);
    const [history, setHistory] = useState([
        chalk.dim(`ready — ${config.model.split('/').pop()} • effort=${config.effort} • target=${target}`),
        chalk.dim('Type text to humanize, / for commands'),
    ]);
    const [lastResult, setLastResult] = useState(null);
    const isSlash = input.startsWith('/');
    useInput((char, key) => {
        if (busy)
            return; // block input while Python is generating
        // tab autocomplete in slash mode
        if (key.tab && isSlash) {
            const cmds = ['/help', '/effort', '/model', '/target', '/analyze', '/save', '/quit'];
            const q = input.split(' ')[0].toLowerCase();
            const hit = cmds.find(c => c.startsWith(q));
            if (hit && hit !== q) {
                setInput(hit + ' ');
                return;
            }
        }
        if (key.escape) {
            setInput('');
            return;
        }
        if (key.return) {
            const raw = input.trim();
            if (!raw)
                return;
            if (raw === '/quit' || raw === '/q' || raw === 'exit') {
                exit();
                return;
            }
            if (raw === '/help') {
                setHistory(s => [...s, chalk.dim('› /help'), chalk.dim('  /effort [quick|standard|deep|max]  /model [id]  /target 0.85  /analyze <text>')]);
                setInput('');
                return;
            }
            if (raw.startsWith('/')) {
                setHistory(s => [...s, chalk.cyan(`› ${raw}`), chalk.dim('  (command — use wizard at start or restart with --effort/--model)')]);
                setInput('');
                return;
            }
            // real humanize via Python bridge (no echo — pipeline rejects near-copies)
            setBusy(true);
            setHistory(s => [...s, chalk.cyan(`› ${raw.slice(0, 64)}${raw.length > 64 ? '…' : ''}`), chalk.dim('⚙ analyze → ✎ generate → ✓ validate …')]);
            setInput('');
            humanizeViaPython(raw, config.effort, config.model).then(res => {
                if (res.ok) {
                    setHistory(s => [...s.slice(0, -1), chalk.dim('⚙ analyze → ✎ generate → ✓ validate'), chalk.whiteBright(res.text)]);
                    setLastResult({ score: res.score, sim: res.sim });
                }
                else {
                    setHistory(s => [...s.slice(0, -1), chalk.dim('⚙ analyze → ✎ generate → ✗ failed'), chalk.red(`  ${res.error}`)]);
                }
                setBusy(false);
            });
            return;
        }
        if (key.backspace || key.delete) {
            setInput(s => s.slice(0, -1));
            return;
        }
        if (key.ctrl && char === 'c') {
            exit();
            return;
        }
        if (char && !key.ctrl && !key.meta)
            setInput(s => s + char);
    });
    const shown = useMemo(() => history.slice(-10), [history]);
    return (_jsxs(Box, { flexDirection: "column", padding: 1, children: [_jsxs(Box, { borderStyle: "round", borderColor: "#2A2E3A", paddingX: 1, justifyContent: "space-between", children: [_jsxs(Box, { children: [_jsx(Text, { bold: true, color: "white", children: "humaize" }), _jsx(Text, { dimColor: true, children: "  local AI-pattern humanizer" })] }), _jsxs(Text, { dimColor: true, children: [config.effort, " \u2022 ", config.model.split('/').pop()] })] }), _jsx(Box, { flexDirection: "column", marginTop: 1, borderStyle: "single", borderColor: "gray", paddingX: 1, paddingY: 0, minHeight: 10, children: shown.map((l, i) => (_jsx(Text, { children: l }, i))) }), lastResult && (_jsxs(Box, { marginTop: 1, borderStyle: "round", borderColor: "gray", paddingX: 1, children: [_jsx(ScoreBar, { score: lastResult.score, target: target }), _jsxs(Text, { dimColor: true, children: ["  sim ", lastResult.sim.toFixed(2)] })] })), _jsxs(Box, { marginTop: 1, children: [_jsxs(Text, { color: "cyan", children: [busy ? '…' : 'you', " \u203A "] }), _jsx(Text, { color: busy ? 'gray' : isSlash ? 'yellow' : 'whiteBright', children: busy ? chalk.dim('humanizing…') : (input || chalk.dim(isSlash ? 'type a command…' : 'type to humanize…')) }), !busy && _jsx(Text, { dimColor: true, children: "\u2588" })] }), _jsx(SlashOverlay, { input: input }), !isSlash && _jsx(Box, { marginTop: 1, children: _jsx(Text, { dimColor: true, children: "  / for commands \u2022 enter to send \u2022 esc clear \u2022 ctrl+c quit" }) })] }));
}
