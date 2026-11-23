export function createHand(
  container: HTMLElement,
  options: { mode: string; background: boolean; numbers: boolean; controls: boolean },
): {
  ready: Promise<boolean>
  setPose: (pose: unknown, options: { instant: boolean }) => void
  playMotion: (pose: unknown) => Promise<boolean>
  dispose: () => void
}
