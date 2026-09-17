import * as p from '@clack/prompts';
import chalk from 'chalk';
import boxen from 'boxen';
import { theme, EFFORT_HINT } from './theme.js';

export type WizardResult = {
  model: string;
  effort: string;
};

const MODELS = [
  { value: 'HuggingFaceTB/SmolLM2-360M-Instruct', label: 'SmolLM2-360M', hint: 'CPU • fast • default' },
  { value: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', label: 'SmolLM2-1.7B', hint: 'CPU • better' },
  { value: 'Qwen/Qwen2.5-7B-Instruct', label: 'Qwen2.5-7B', hint: 'GPU 4-bit • best' },
  { value: 'mistralai/Mistral-7B-Instruct-v0.3', label: 'Mistral-7B', hint: 'GPU 4-bit • alt' },
] as const;

const EFFORTS = [
  { value: 'quick', label: 'Quick' },
  { value: 'standard', label: 'Standard' },
  { value: 'deep', label: 'Deep' },
  { value: 'max', label: 'Max' },
] as const;

export async function runWizard(): Promise<WizardResult | null> {
  console.clear();

  // Kilo-style intro: bold white + dim subtitle, single weight
  p.intro(`${chalk.whiteBright.bold(' humaize ')}${chalk.dim('— local AI-pattern humanizer')}`);

  const result = await p.group(
    {
      model: () =>
        p.select({
          message: 'Select model',
          options: MODELS.map(m => ({ value: m.value, label: m.label, hint: m.hint })),
        }),
      effort: () =>
        p.select({
          message: 'Effort level',
          options: EFFORTS.map(e => ({
            value: e.value,
            label: e.label,
            hint: EFFORT_HINT[e.value],
          })),
        }),
    },
    {
      onCancel: () => {
        p.cancel('Setup cancelled.');
        process.exit(0);
      },
    }
  );

  // boxen status panel — single-line rounded, dim border (mirrors Kilo card)
  const summary = `${chalk.dim('model:')} ${chalk.whiteBright(result.model.split('/').pop())}  ${chalk.dim('•')}  ${chalk.dim('effort:')} ${chalk.whiteBright(result.effort)}  ${chalk.dim(`— ${EFFORT_HINT[result.effort as string]}`)}`;
  console.log(
    boxen(summary, {
      padding: { left: 1, right: 1, top: 0, bottom: 0 },
      margin: { top: 1, bottom: 0, left: 0, right: 0 },
      borderStyle: 'round',
      borderColor: 'gray',
      dimBorder: true,
    })
  );

  const s = p.spinner();
  s.start('Warming local pipeline…');
  await new Promise(r => setTimeout(r, 650));
  s.stop(chalk.dim('Ready — launching dashboard'));

  // tiny pause for the eye to register the transition
  await new Promise(r => setTimeout(r, 300));

  return result as WizardResult;
}
