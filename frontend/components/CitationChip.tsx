// CitationChip.tsx - the small [n] button inside an answer.
// Click it and the matching source is highlighted below the answer.

interface Props {
  number: number;
  valid: boolean; // false when the model cited a source number that does not exist
  active: boolean;
  onSelect: (n: number) => void;
}

export default function CitationChip({ number, valid, active, onSelect }: Props) {
  if (!valid) {
    return (
      <span title="This citation points to a source that does not exist" className="mx-0.5 rounded bg-red-100 px-1.5 text-xs font-semibold text-red-700">
        {number}
      </span>
    );
  }
  return (
    <button
      type="button"
      onClick={() => onSelect(number)}
      title={`Show source ${number}`}
      className={`mx-0.5 rounded px-1.5 text-xs font-semibold transition-colors ${
        active ? "bg-emerald-700 text-white" : "bg-emerald-100 text-emerald-800 hover:bg-emerald-200"
      }`}
    >
      {number}
    </button>
  );
}
