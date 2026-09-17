import React, { useState, useMemo } from 'react';
import { Box, Text, useInput, useApp } from 'ink';
import chalk from 'chalk';
import { SlashOverlay } from './SlashOverlay.js';
import { humanizeViaPython } from '../bridge.js';

type DashboardProps = {
  config: { model: string; effort: string };
};

function ScoreBar({ score, target }: { score: number; target: number }) {
  const met = score >= target;
  const width = 24;
  const fill = Math.round(Math.min(1, Math.max(0, score)) * width);
  const bar = '█'.repeat(fill) + '░'.repeat(width - fill);
  const color = met ? chalk.hex('#5FBF7A') : score > 0.4 ? chalk.hex('#E0A83C') : chalk.red;
  return (
    <Text>
      {color(`[${bar}]`)} {chalk.whiteBright(`${score.toFixed(3)} / ${target.toFixed(2)}`)} {met ? chalk.green('✓ PASS') : chalk.yellow('✗ retry')}
    </Text>
  );
}

export function Dashboard({ config }: DashboardProps) {
  const { exit } = useApp();
  const [target] = useState(0.85);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<string[]>([
    chalk.dim(`ready — ${config.model.split('/').pop()} • effort=${config.effort} • target=${target}`),
    chalk.dim('Type text to humanize, / for commands'),
  ]);
  const [lastResult, setLastResult] = useState<{ score: number; sim: number } | null>(null);

  const isSlash = input.startsWith('/');

  useInput((char, key) => {
    if (busy) return; // block input while Python is generating
    // tab autocomplete in slash mode
    if (key.tab && isSlash) {
      const cmds = ['/help', '/effort', '/model', '/target', '/analyze', '/save', '/quit'];
      const q = input.split(' ')[0]!.toLowerCase();
      const hit = cmds.find(c => c.startsWith(q));
      if (hit && hit !== q) { setInput(hit + ' '); return; }
    }
    if (key.escape) { setInput(''); return; }
    if (key.return) {
      const raw = input.trim();
      if (!raw) return;
      if (raw === '/quit' || raw === '/q' || raw === 'exit') { exit(); return; }
      if (raw === '/help') {
        setHistory(s => [...s, chalk.dim('› /help'), chalk.dim('  /effort [quick|standard|deep|max]  /model [id]  /target 0.85  /analyze <text>')]);
        setInput(''); return;
      }
      if (raw.startsWith('/')) {
        setHistory(s => [...s, chalk.cyan(`› ${raw}`), chalk.dim('  (command — use wizard at start or restart with --effort/--model)')]);
        setInput(''); return;
      }
      // real humanize via Python bridge (no echo — pipeline rejects near-copies)
      setBusy(true);
      setHistory(s => [...s, chalk.cyan(`› ${raw.slice(0, 64)}${raw.length > 64 ? '…' : ''}`), chalk.dim('⚙ analyze → ✎ generate → ✓ validate …')]);
      setInput('');
      humanizeViaPython(raw, config.effort, config.model).then(res => {
        if (res.ok) {
          setHistory(s => [...s.slice(0, -1), chalk.dim('⚙ analyze → ✎ generate → ✓ validate'), chalk.whiteBright(res.text)]);
          setLastResult({ score: res.score, sim: res.sim });
        } else {
          setHistory(s => [...s.slice(0, -1), chalk.dim('⚙ analyze → ✎ generate → ✗ failed'), chalk.red(`  ${res.error}`)]);
        }
        setBusy(false);
      });
      return;
    }
    if (key.backspace || key.delete) { setInput(s => s.slice(0, -1)); return; }
    if (key.ctrl && char === 'c') { exit(); return; }
    if (char && !key.ctrl && !key.meta) setInput(s => s + char);
  });

  const shown = useMemo(() => history.slice(-10), [history]);

  return (
    <Box flexDirection="column" padding={1}>
      {/* Header — Kilo style: single line, left title + right meta dim */}
      <Box borderStyle="round" borderColor="#2A2E3A" paddingX={1} justifyContent="space-between">
        <Box>
          <Text bold color="white">humaize</Text>
          <Text dimColor>  local AI-pattern humanizer</Text>
        </Box>
        <Text dimColor>{config.effort} • {config.model.split('/').pop()}</Text>
      </Box>

      {/* Log */}
      <Box
        flexDirection="column"
        marginTop={1}
        borderStyle="single"
        borderColor="gray"
        paddingX={1}
        paddingY={0}
        minHeight={10}
      >
        {shown.map((l, i) => (
          <Text key={i}>{l}</Text>
        ))}
      </Box>

      {/* Score bar — boxen-style single rounded panel */}
      {lastResult && (
        <Box marginTop={1} borderStyle="round" borderColor="gray" paddingX={1}>
          <ScoreBar score={lastResult.score} target={target} />
          <Text dimColor>  sim {lastResult.sim.toFixed(2)}</Text>
        </Box>
      )}

      {/* Input */}
      <Box marginTop={1}>
        <Text color="cyan">{busy ? '…' : 'you'} › </Text>
        <Text color={busy ? 'gray' : isSlash ? 'yellow' : 'whiteBright'}>{busy ? chalk.dim('humanizing…') : (input || chalk.dim(isSlash ? 'type a command…' : 'type to humanize…'))}</Text>
        {!busy && <Text dimColor>█</Text>}
      </Box>

      {/* Slash type-ahead overlay — only when input starts with / */}
      <SlashOverlay input={input} />

      {!isSlash && <Box marginTop={1}><Text dimColor>  / for commands • enter to send • esc clear • ctrl+c quit</Text></Box>}
    </Box>
  );
}
