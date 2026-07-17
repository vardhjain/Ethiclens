import { Alert, Button, Center, Stack, Text, Title } from "@mantine/core";
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

// Class component is required here — React has no hook equivalent for
// getDerivedStateFromError/componentDidCatch as of React 18.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Unhandled render error:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <Center py="xl" px="md">
          <Stack maw={480} gap="sm">
            <Title order={3}>Something went wrong</Title>
            <Alert color="red" title="Unexpected error">
              <Text size="sm">
                {error.message || "The app hit an unexpected error."}
              </Text>
            </Alert>
            <Button
              onClick={() => window.location.reload()}
              style={{ alignSelf: "flex-start" }}
            >
              Reload
            </Button>
          </Stack>
        </Center>
      );
    }
    return this.props.children;
  }
}
