import { Alert, Button, Stack, Text } from "@mantine/core";

interface QueryErrorProps {
  error: unknown;
  onRetry: () => void;
  title?: string;
}

export function QueryError({ error, onRetry, title = "Couldn't load this data" }: QueryErrorProps) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <Alert color="red" title={title}>
      <Stack gap="xs">
        <Text size="sm">{message}</Text>
        <Button size="xs" variant="light" onClick={onRetry} style={{ alignSelf: "flex-start" }}>
          Retry
        </Button>
      </Stack>
    </Alert>
  );
}
