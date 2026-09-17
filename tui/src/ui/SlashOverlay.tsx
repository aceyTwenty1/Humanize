import React from 'react';
import { Box, Text } from 'ink';
import chalk from 'chalk';

type Props = { input: string; onSelect?: (cmd: string) => void };

const COMMANDS: { cmd: string; desc: string }[] = [
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

export function SlashOverlay({ input }: Props) {
  if (!input.startsWith('/')) return null;

  const q = input.toLowerCase().split(' ')[0]!;
  const filtered = COMMANDS.filter(c => c.cmd.startsWith(q));

  // exact typing dim, matched prefix white
  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor="gray"
      paddingX={1}
      paddingY={0}
      marginTop={1}
    >
      <Box marginBottom={1}>
        <Text dimColor>{chalk.dim('commands')} </Text>
        <Text color="gray">{chalk.dim(`— ${filtered.length} matches`)}</Text>
      </Box>

      {filtered.length ? (
        filtered.map(c => {
          const isActive = c.cmd === q;
          return (
            <Box key={c.cmd} justifyContent="space-between">
              <Text>
                {isActive ? chalk.whiteBright.bold(`› ${c.cmd}`) : chalk.dim(`  ${c.cmd}`)}
                <Text dimColor>{`  ${chalk.dim(c.desc)}`}</Text>
              </Text>
            </Box>
          );
        })
      ) : (
        <Text color="yellow">  no match — try /help</Text>
      )}

      <Box marginTop={1}>
        <Text dimColor>{chalk.dim('↵ select  •  esc dismiss  •  tab autocomplete')}</Text>
      </Box>
    </Box>
  );
}
