import chalk from 'chalk';
// Kilo Code micro-palette — dim metadata, white active, muted borders
export const theme = {
    dim: chalk.dim,
    active: chalk.whiteBright.bold,
    success: chalk.hex('#5FBF7A'),
    warning: chalk.hex('#E0A83C'),
    error: chalk.hex('#DE6B6B'),
    cyan: chalk.cyan,
    accent: chalk.hex('#7AA5FF'),
    border: '#2A2E3A',
    borderDim: 'gray',
};
export const EFFORT_HINT = {
    quick: '2 iters • no polish — fastest',
    standard: '4 iters + 1 polish — balanced',
    deep: '6 iters • 2 cand • 2 polish',
    max: '8 iters • relaxed floor — max',
};
