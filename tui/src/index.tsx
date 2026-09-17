#!/usr/bin/env node
/**
 * Humaize TUI — orchestrator
 * 1. Clack wizard (step-by-step, vertical connectors)
 * 2. → clear → Ink dashboard loop (react, splits, slash type-ahead)
 *
 * The handoff is `await wizard()` then `render(<Dashboard>)` — stdin is
 * fully released by clack before ink's useInput mounts, so no reader clash.
 */
import { render } from 'ink';
import React from 'react';
import { runWizard } from './wizard.js';
import { Dashboard } from './ui/Dashboard.js';

async function main() {
  // --no-wizard / --model / --effort flags skip clack for one-shot runs
  const args = process.argv.slice(2);
  const noWizard = args.includes('--no-wizard');
  const effortArg = args.find(a => a.startsWith('--effort='))?.split('=')[1];
  const modelArg = args.find(a => a.startsWith('--model='))?.split('=')[1];

  let config: { model: string; effort: string };

  if (noWizard || (effortArg && modelArg)) {
    config = {
      model: modelArg ?? 'HuggingFaceTB/SmolLM2-360M-Instruct',
      effort: effortArg ?? 'standard',
    };
  } else {
    const wizardResult = await runWizard();
    if (!wizardResult) process.exit(0);
    config = wizardResult;
  }

  // critical transition: clear clack's rendered lines before ink mounts
  console.clear();

  const { waitUntilExit } = render(React.createElement(Dashboard, { config }));
  await waitUntilExit();
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
