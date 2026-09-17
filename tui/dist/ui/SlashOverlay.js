import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import { Box, Text } from 'ink';
import chalk from 'chalk';
const COMMANDS = [
    { cmd: '/help', desc: 'show commands' },
    { cmd: '/effort', desc: 'switch effort (quick/standard/deep/max)' },
    { cmd: '/model', desc: 'switch generator model' },
    { cmd: '/target', desc: 'set goal 0.5–0.95' },
    { cmd: '/iters', desc: 'max retries' },
    { cmd: '/sim', desc: 'min input overlap' },
    { cmd: '/polish', desc: 'refinement passes' },
    { cmd: '/analyze', desc: 'score only — no rewrite' },
    { cmd: '/save', desc: 'save last rewrite' },
    { cmd: '/quit', desc: 'exit' },
];
export function SlashOverlay({ input }) {
    if (!input.startsWith('/'))
        return null;
    const q = input.toLowerCase().split(' ')[0];
    const filtered = COMMANDS.filter(c => c.cmd.startsWith(q));
    // exact typing dim, matched prefix white
    return (_jsxs(Box, { flexDirection: "column", borderStyle: "round", borderColor: "gray", paddingX: 1, paddingY: 0, marginTop: 1, children: [_jsxs(Box, { marginBottom: 1, children: [_jsxs(Text, { dimColor: true, children: [chalk.dim('commands'), " "] }), _jsx(Text, { color: "gray", children: chalk.dim(`— ${filtered.length} matches`) })] }), filtered.length ? (filtered.map(c => {
                const isActive = c.cmd === q;
                return (_jsx(Box, { justifyContent: "space-between", children: _jsxs(Text, { children: [isActive ? chalk.whiteBright.bold(`› ${c.cmd}`) : chalk.dim(`  ${c.cmd}`), _jsx(Text, { dimColor: true, children: `  ${chalk.dim(c.desc)}` })] }) }, c.cmd));
            })) : (_jsx(Text, { color: "yellow", children: "  no match \u2014 try /help" })), _jsx(Box, { marginTop: 1, children: _jsx(Text, { dimColor: true, children: chalk.dim('↵ select  •  esc dismiss  •  tab autocomplete') }) })] }));
}
