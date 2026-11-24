// SafetyBanner.tsx - a coloured box for emergency / health / withheld messages.

interface Props {
  tone: "emergency" | "health" | "info";
  children: React.ReactNode;
}

const STYLES: Record<Props["tone"], string> = {
  emergency: "border-red-300 bg-red-50 text-red-900",
  health: "border-amber-300 bg-amber-50 text-amber-900",
  info: "border-emerald-200 bg-emerald-50 text-emerald-900",
};

export default function SafetyBanner({ tone, children }: Props) {
  return (
    <div role={tone === "emergency" ? "alert" : "note"} className={`rounded-lg border px-3 py-2 text-sm ${STYLES[tone]}`}>
      {children}
    </div>
  );
}
