import {
  Alert,
  Badge,
  Button,
  Card,
  Group,
  Loader,
  Stack,
  Table,
  Text,
  Title,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useNavigate, useParams } from "react-router-dom";
import { api, getToken, type MetricOut } from "../api/client";

function diChartData(metrics: MetricOut[]) {
  return metrics
    .filter((m) => m.metric_type === "disparate_impact" && m.value !== null)
    .map((m) => ({
      group: m.group_label,
      di: m.value as number,
      err: [
        (m.value as number) - (m.ci_low ?? (m.value as number)),
        (m.ci_high ?? (m.value as number)) - (m.value as number),
      ] as [number, number],
      flagged: (m.value as number) < 0.8,
    }));
}

async function downloadReport(id: string) {
  const res = await fetch(api.reportUrl(id), {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!res.ok) throw new Error("Report not ready");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `EthicLens_Scorecard_${id}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

export function SessionDetailPage() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const qc = useQueryClient();

  const session = useQuery({
    queryKey: ["session", id],
    queryFn: () => api.getSession(id),
    refetchInterval: (q) =>
      ["RUNNING", "QUEUED"].includes(q.state.data?.status ?? "") ? 2000 : false,
  });
  const done = ["COMPLETED", "FLAGGED", "SIGNED_OFF"].includes(session.data?.status ?? "");
  const metrics = useQuery({
    queryKey: ["metrics", id],
    queryFn: () => api.metrics(id),
    enabled: done,
  });
  const recs = useQuery({
    queryKey: ["recs", id],
    queryFn: () => api.recommendations(id),
    enabled: done,
  });

  const mitigate = useMutation({
    mutationFn: (strategy: string) => api.mitigate(id, strategy),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["sessions"] });
      if (r.result_session_id) {
        notifications.show({ color: "green", message: "Mitigation applied — re-audited." });
        nav(`/sessions/${r.result_session_id}`);
      }
    },
    onError: (e) => notifications.show({ color: "red", message: (e as Error).message }),
  });

  if (session.isLoading || !session.data) return <Loader />;
  const s = session.data;
  const flaggedRecs = recs.data?.recommendations ?? {};
  const firstGroup = Object.keys(flaggedRecs)[0];

  return (
    <Stack>
      <Group justify="space-between">
        <Title order={3}>{s.name}</Title>
        <Group>
          <Badge color={s.status === "FLAGGED" ? "red" : "green"}>{s.status}</Badge>
          {s.composite_band && <Badge variant="light">{s.composite_band}</Badge>}
          {done && (
            <Button size="xs" variant="light" onClick={() => downloadReport(id).catch(() => {})}>
              Download Scorecard PDF
            </Button>
          )}
        </Group>
      </Group>

      {!done && <Alert color="yellow">Audit in progress… this page updates automatically.</Alert>}

      {done && metrics.data && (
        <>
          <Card withBorder>
            <Text fw={600} mb="sm">
              Disparate Impact by group (95% CI; red = below the 0.80 four-fifths rule)
            </Text>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={diChartData(metrics.data.metrics)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="group" />
                <YAxis domain={[0, 1.2]} />
                <Tooltip />
                <ReferenceLine y={0.8} stroke="#34495e" strokeDasharray="4 4" />
                <Bar dataKey="di">
                  <ErrorBar dataKey="err" width={4} stroke="#555" />
                  {diChartData(metrics.data.metrics).map((d, i) => (
                    <Cell key={i} fill={d.flagged ? "#c0392b" : "#27ae60"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card withBorder>
            <Table>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Group</Table.Th>
                  <Table.Th>Metric</Table.Th>
                  <Table.Th>Value</Table.Th>
                  <Table.Th>95% CI</Table.Th>
                  <Table.Th>Status</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {metrics.data.metrics.map((m, i) => (
                  <Table.Tr key={i}>
                    <Table.Td>{m.group_label}</Table.Td>
                    <Table.Td>{m.metric_type}</Table.Td>
                    <Table.Td>{m.value?.toFixed(3) ?? "N/A"}</Table.Td>
                    <Table.Td>
                      {m.ci_low != null ? `[${m.ci_low.toFixed(2)}, ${m.ci_high?.toFixed(2)}]` : "—"}
                    </Table.Td>
                    <Table.Td>{m.classification ?? "—"}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Card>
        </>
      )}

      {firstGroup && (
        <Card withBorder>
          <Text fw={600} mb="sm">
            Recommended mitigations for {firstGroup}
          </Text>
          <Stack gap="xs">
            {flaggedRecs[firstGroup].map((r) => (
              <Group key={r.rank} justify="space-between">
                <Text size="sm">
                  #{r.rank} {r.strategy_name} (proj. +{r.estimated_di_improvement.toFixed(2)} DI,{" "}
                  {r.stage})
                </Text>
                {r.rank === 1 && (
                  <Button
                    size="xs"
                    loading={mitigate.isPending}
                    onClick={() => mitigate.mutate(r.strategy)}
                  >
                    Apply &amp; re-audit
                  </Button>
                )}
              </Group>
            ))}
          </Stack>
        </Card>
      )}
    </Stack>
  );
}
