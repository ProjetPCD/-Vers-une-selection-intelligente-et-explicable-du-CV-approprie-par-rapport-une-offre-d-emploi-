import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { CVRankItem } from '../../services/cv-matcher.service';

@Component({
  selector: 'app-results-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results-table.html',
  styleUrl: './results-table.scss'
})
export class ResultsTableComponent {
  @Input() rankings: CVRankItem[] = [];

  tierColor(tier: string | null): string {
    if (!tier) return '#8b89a8';
    const t = tier.toUpperCase();
    if (t.includes('ELITE')) return '#22c55e';
    if (t.includes('STRONG')) return '#84cc16';
    if (t.includes('GOOD')) return '#facc15';
    if (t.includes('WEAK')) return '#f97316';
    if (t.includes('POOR') || t.includes('REJECT')) return '#ef4444';
    return '#8b89a8';
  }

  verdictIcon(verdict: string | null): string {
    if (!verdict) return '—';
    const v = verdict.toUpperCase();
    if (v.includes('STRONG') || v.includes('ELITE')) return '✅';
    if (v.includes('GOOD')) return '🟡';
    if (v.includes('WEAK')) return '🟠';
    return '❌';
  }

  pct(val: number | null): string {
    return val != null ? val.toFixed(1) + '%' : '—';
  }

  score(val: number | null): string {
    return val != null ? val.toFixed(3) : '—';
  }

  barWidth(val: number | null): number {
    return val != null ? Math.min(val, 100) : 0;
  }
}