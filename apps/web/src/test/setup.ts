import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement matchMedia; MantineProvider calls it to resolve
// defaultColorScheme="auto" against the OS preference.
if (!window.matchMedia) {
  window.matchMedia = (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  });
}
