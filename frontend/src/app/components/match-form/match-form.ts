import { Component, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  CvMatcherService,
  CVRankItem,
  StatusResponse,
  OutputFile,
  AvailableCv
} from '../../services/cv-matcher.service';
import { ResultsTableComponent } from '../results-table/results-table';
import { GraphViewerComponent } from '../graph-viewer/graph-viewer';

@Component({
  selector: 'app-match-form',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultsTableComponent, GraphViewerComponent],
  templateUrl: './match-form.html',
  styleUrl: './match-form.scss'
})
export class MatchFormComponent {

  @ViewChild('fileInput') fileInputRef!: ElementRef<HTMLInputElement>;

  jdText = '';
  selectedFiles: File[] = [];
  state: 'idle' | 'uploading' | 'processing' | 'done' | 'error' = 'idle';
  jobId = '';
  statusMsg = '';
  currentPhase = '';
  progressMsg = '';
  rankings: CVRankItem[] = [];
  errorMsg = '';
  charCount = 0;
  summaryMsg = '';

  graphData: any = null;
  phase2Data: any = null;
  phase3Data: any = null;
  reportData: any = null;
  outputFiles: OutputFile[] = [];

  liveCvs: AvailableCv[] = [];
  liveReports: Record<number, any> = {};

  activeTab: 'overview' | 'graph' | 'phase2' | 'phase3' | 'report' | 'outputs' = 'overview';
  selectedCvIndex = 0;

  // ── Retry ────────────────────────────────────────────────────────────────
  private _tentativesResultats = 0;
  private readonly MAX_TENTATIVES = 30;
  private _retryTimerId: any = null;
  private _safetyTimerId: any = null;
  private _stopPolling: (() => void) | null = null; // ← stop function du polling

  // ── Computed ─────────────────────────────────────────────────────────────

  get hasLiveData(): boolean {
    return !!(this.graphData || this.liveCvsWithReport.length > 0);
  }

  get phase2Graph(): any {
    if (!this.phase2Data) return null;
    const noeuds = Object.entries(this.phase2Data.skill_instructions || {}).map(([id, instr]: [string, any]) => ({
      id, label: id,
      type: instr.skill_type || 'Hard Skill',
      w_0: instr.evidence_verdict === 'strong' ? 1.0
        : instr.evidence_verdict === 'moderate' ? 0.6 : 0.3
    }));
    const aretes = Object.entries(this.phase2Data.edge_instructions || {}).map(([_k, instr]: [string, any]) => ({
      source: instr.source, target: instr.target,
      relation: instr.synergy_found ? 'synergie' : 'lié'
    }));
    return { nodes: noeuds, edges: aretes };
  }

  get liveCvsWithReport(): AvailableCv[] {
    return this.liveCvs.filter(cv => cv.has_report);
  }

  get showResultsTabs(): boolean {
    return this.state === 'done' || (this.state === 'processing' && this.hasLiveData);
  }

  constructor(private svc: CvMatcherService) { }

  // ── Fichiers ──────────────────────────────────────────────────────────────

  openFilePicker() { this.fileInputRef?.nativeElement?.click(); }

  onFilesChange(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const nvx = Array.from(input.files).filter(
        f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf')
      );
      this.selectedFiles = [...this.selectedFiles, ...nvx];
      input.value = '';
    }
  }

  removeFile(i: number) {
    this.selectedFiles = this.selectedFiles.filter((_, idx) => idx !== i);
  }

  canSubmit(): boolean {
    return this.charCount > 20 && this.selectedFiles.length > 0;
  }

  // ── Soumission ────────────────────────────────────────────────────────────

  submit() {
    const jd = this.jdText.trim();
    if (jd.length < 20 || this.selectedFiles.length === 0) return;

    this._cancelRetry();
    if (this._stopPolling) { this._stopPolling(); this._stopPolling = null; }
    this.state = 'uploading';
    this.errorMsg = '';
    this.rankings = [];
    this.graphData = null;
    this.phase2Data = null;
    this.phase3Data = null;
    this.reportData = null;
    this.outputFiles = [];
    this.liveCvs = [];
    this.liveReports = {};
    this._tentativesResultats = 0;

    this.svc.submitMatch(jd, this.selectedFiles).subscribe({
      next: res => {
        this.jobId = res.job_id;
        this.state = 'processing';
        this.statusMsg = 'Pipeline en cours…';
        this.startPolling();
      },
      error: err => {
        this.state = 'error';
        this.errorMsg = err?.error?.detail || 'Erreur lors de la soumission';
      }
    });
  }

  // ── Forçage manuel (bouton "Actualiser") ──────────────────────────────────

  forceLoadResults() {
    this.svc.getStatus(this.jobId).subscribe({
      next: (res: StatusResponse) => {
        if (res.status === 'done') {
          this.state = 'done';
          this.activeTab = 'overview';
          this.statusMsg = '✅ Pipeline terminé !';
          this.summaryMsg = res.summary || '';
          this._cancelRetry();
          this.loadResultsWithRetry();
          this.loadAllPhaseData();
        } else {
          this.statusMsg = `⏳ Encore en cours : ${res.current_phase || res.status}`;
        }
      },
      error: () => { this.statusMsg = '⚠️ Impossible de vérifier le statut'; }
    });
  }

  // ── Polling du statut ─────────────────────────────────────────────────────

  startPolling() {
    // ✅ FIX SIMPLE : vérification forcée toutes les 5s indépendamment du polling
    // Cela garantit qu'on détecte "done" même si le polling a une anomalie
    const forceCheckInterval = setInterval(() => {
      if (this.state === 'done' || this.state === 'error' || this.state === 'idle') {
        clearInterval(forceCheckInterval);
        return;
      }
      this.svc.getStatus(this.jobId).subscribe({
        next: (res: StatusResponse) => {
          if (res.status === 'done' && this.state === 'processing') {
            console.log('[ForceCheck] ✅ status=done détecté — bascule immédiate');
            clearInterval(forceCheckInterval);
            this.state = 'done';
            this.activeTab = 'overview';
            this.statusMsg = '✅ Pipeline terminé !';
            this.summaryMsg = res.summary || '';
            this.loadResultsWithRetry();
            this.loadAllPhaseData();
          }
        },
        error: () => { }
      });
    }, 5000);

    // Stocker pour nettoyage au reset
    this._safetyTimerId = forceCheckInterval as any;

    this._stopPolling = this.svc.startPollingLoop(
      this.jobId,

      // onStatus — appelé à chaque tick tant que pas done/error
      (res: StatusResponse) => {
        this.currentPhase = res.current_phase || '';
        this.progressMsg = res.progress || '';
        this.summaryMsg = res.summary || '';
        this.statusMsg = `⏳ ${res.current_phase || res.status}…`;
        this.tryLoadProgressiveData();
      },

      // onDone — appelé UNE SEULE FOIS quand status === 'done' ou 'error'
      (res: StatusResponse) => {
        this.currentPhase = res.current_phase || '';
        this.progressMsg = res.progress || '';
        this.summaryMsg = res.summary || '';

        if (res.status === 'done') {
          // ✅ FIX PRINCIPAL : on bascule IMMÉDIATEMENT à 'done'
          // sans attendre le chargement des résultats
          this.state = 'done';
          this.activeTab = 'overview';
          this.statusMsg = '✅ Pipeline terminé !';
          this._tentativesResultats = 0;
          this._cancelRetry();
          this.loadResultsWithRetry();
          this.loadAllPhaseData();

        } else {
          // status === 'error'
          this.state = 'error';
          this.errorMsg = res.progress || 'Le pipeline a rencontré une erreur.';
        }
      },

      // onFatalError — trop d'erreurs réseau consécutives
      () => {
        this.state = 'error';
        this.errorMsg = 'Connexion perdue. Vérifiez que le backend tourne sur localhost:8000.';
      }
    );
  }

  // ── Données progressives ──────────────────────────────────────────────────

  private tryLoadProgressiveData() {
    if (!this.graphData) {
      this.svc.getGraph(this.jobId).subscribe({
        next: data => {
          if (data && !data.status) {
            this.graphData = data;
            if (this.activeTab === 'overview') this.activeTab = 'graph';
          }
        },
        error: () => { }
      });
    }
    this.svc.getAvailable(this.jobId).subscribe({
      next: avail => {
        this.liveCvs = avail.cvs || [];
        for (const cv of this.liveCvs) {
          if (cv.has_report && this.liveReports[cv.cv_index] === undefined) {
            this.svc.getReport(this.jobId, cv.cv_index).subscribe({
              next: data => { if (data && !data.status) this.liveReports[cv.cv_index] = data; },
              error: () => { }
            });
          }
        }
      },
      error: () => { }
    });
  }

  // ── Chargement résultats — ROBUSTE ────────────────────────────────────────
  //
  // Stratégie :
  //   - On appelle /api/results dès que status="done"
  //   - Si 202/404/0 → on réessaie avec délai progressif (max 30 fois)
  //   - JAMAIS on ne reste bloqué à "processing" après un "done" backend
  //   - Si toutes les tentatives échouent → on affiche ce qu'on a déjà

  loadResultsWithRetry() {
    this.svc.getResults(this.jobId).subscribe({
      next: res => {
        if (res.rankings && res.rankings.length > 0) {
          // ✅ SUCCÈS
          this.rankings = res.rankings;
          this.state = 'done';
          this.activeTab = 'overview';
          this.statusMsg = '✅ Terminé !';
          this._cancelRetry(); // annule le safety timer aussi
          console.log(`[Résultats] ✅ ${res.rankings.length} résultat(s) en ${this._tentativesResultats} tentative(s)`);
        } else {
          // 200 OK mais rankings vide → MongoDB pas encore prêt
          this._scheduleResultsRetry(0);
        }
      },
      error: err => {
        const http = err?.status;
        const retryable = http === 202 || http === 404 || http === 0 || http === 500;
        if (retryable && this._tentativesResultats < this.MAX_TENTATIVES) {
          this._scheduleResultsRetry(http);
        } else {
          console.error(`[Résultats] ❌ Abandon après ${this._tentativesResultats} tentatives (HTTP ${http})`);
          this.state = 'done';
          this._cancelRetry();
          if (this.graphData) this.activeTab = 'graph';
          else if (Object.keys(this.liveReports).length > 0) this.activeTab = 'report';
          else this.activeTab = 'overview';
        }
      }
    });
  }

  private _scheduleResultsRetry(http: number) {
    this._tentativesResultats++;
    let delai: number;
    if (this._tentativesResultats <= 5) delai = 1000;
    else if (this._tentativesResultats <= 10) delai = 3000;
    else delai = 5000;
    console.warn(`[Résultats] HTTP ${http || 'vide'} — retry ${this._tentativesResultats}/${this.MAX_TENTATIVES} dans ${delai}ms`);
    this._retryTimerId = setTimeout(() => this.loadResultsWithRetry(), delai);
  }

  private _cancelRetry() {
    if (this._retryTimerId !== null) {
      clearTimeout(this._retryTimerId);
      this._retryTimerId = null;
    }
    if (this._safetyTimerId !== null) {
      clearInterval(this._safetyTimerId); // c'est un interval maintenant
      this._safetyTimerId = null;
    }
  }

  // ── Chargement phases après "done" ────────────────────────────────────────

  loadAllPhaseData() {
    this.svc.getGraph(this.jobId).subscribe({
      next: data => { if (data && !data.status) this.graphData = data; },
      error: () => { }
    });
    this.svc.getOutputs(this.jobId).subscribe({
      next: res => { this.outputFiles = res.files; },
      error: () => { }
    });
    this.loadCvPhaseData(0);
  }

  loadCvPhaseData(cvIndex: number) {
    this.selectedCvIndex = cvIndex;
    this.phase2Data = null;
    this.phase3Data = null;
    this.reportData = null;

    this.fetchWithRetry(() => this.svc.getPhase2(this.jobId, cvIndex),
      (d: any) => { this.phase2Data = d; }, 5);
    this.fetchWithRetry(() => this.svc.getPhase3(this.jobId, cvIndex),
      (d: any) => { this.phase3Data = d; }, 5);
    this.fetchWithRetry(() => this.svc.getReport(this.jobId, cvIndex),
      (d: any) => { this.reportData = d; }, 5);
  }

  private fetchWithRetry(
    requete: () => any,
    onSucces: (data: any) => void,
    maxTentatives: number,
    tentative = 0
  ) {
    requete().subscribe({
      next: (data: any) => {
        if (data && !data.status) {
          onSucces(data);
        } else if (tentative < maxTentatives) {
          setTimeout(() => this.fetchWithRetry(requete, onSucces, maxTentatives, tentative + 1), 2000);
        }
      },
      error: () => {
        if (tentative < maxTentatives) {
          setTimeout(() => this.fetchWithRetry(requete, onSucces, maxTentatives, tentative + 1), 2000);
        }
      }
    });
  }

  // ── Navigation onglets ────────────────────────────────────────────────────

  setTab(tab: typeof this.activeTab) {
    this.activeTab = tab;
    if (tab === 'phase2' && !this.phase2Data) {
      this.fetchWithRetry(() => this.svc.getPhase2(this.jobId, this.selectedCvIndex),
        (d: any) => { this.phase2Data = d; }, 5);
    }
    if (tab === 'phase3' && !this.phase3Data) {
      this.fetchWithRetry(() => this.svc.getPhase3(this.jobId, this.selectedCvIndex),
        (d: any) => { this.phase3Data = d; }, 5);
    }
    if (tab === 'report' && !this.reportData) {
      this.fetchWithRetry(() => this.svc.getReport(this.jobId, this.selectedCvIndex),
        (d: any) => { this.reportData = d; }, 5);
    }
  }

  downloadFile(file: OutputFile) {
    window.open(this.svc.getOutputFileUrl(file.path), '_blank');
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  getTypeColor(type: string): string {
    const c: Record<string, string> = {
      'Hard Skill': '#1f77b4', 'Soft Skill': '#ff7f0e', 'Responsibility': '#2ca02c',
      'Tool': '#d62728', 'Domain': '#9467bd', 'Experience': '#e377c2',
      'Certification': '#bcbd22', 'Education': '#8c564b'
    };
    return c[type] || '#A9A9A9';
  }

  getEvidenceColor(q: string): string {
    if (!q) return '#8b89a8';
    const ql = q.toLowerCase();
    if (ql.includes('strong')) return '#22c55e';
    if (ql.includes('moderate')) return '#facc15';
    if (ql.includes('weak')) return '#f97316';
    if (ql.includes('absent')) return '#ef4444';
    return '#8b89a8';
  }

  getScoreColor(s: number): string {
    if (s >= 1.5) return '#22c55e';
    if (s >= 1.0) return '#84cc16';
    if (s >= 0.5) return '#facc15';
    if (s >= 0.25) return '#f97316';
    return '#ef4444';
  }

  traduireVerdict(v: string): string {
    const t: Record<string, string> = {
      'strong': 'Solide', 'moderate': 'Modéré', 'weak': 'Faible', 'absent': 'Absent'
    };
    return t[v?.toLowerCase()] || v || '—';
  }

  traduireType(t: string): string {
    const m: Record<string, string> = {
      'Hard Skill': 'Compétence technique', 'Soft Skill': 'Compétence relationnelle',
      'Responsibility': 'Responsabilité', 'Tool': 'Outil', 'Domain': 'Domaine',
      'Experience': 'Expérience', 'Certification': 'Certification', 'Education': 'Formation'
    };
    return m[t] || t || '—';
  }

  traduireIntensite(i: string): string {
    const m: Record<string, string> = {
      'central': 'Centrale', 'secondary': 'Secondaire', 'peripheral': 'Périphérique',
      'high': 'Élevée', 'medium': 'Moyenne', 'low': 'Faible'
    };
    return m[i?.toLowerCase()] || i || '—';
  }

  traduireCategorie(c: string): string {
    const m: Record<string, string> = {
      'inputs': 'Entrées', 'graphs': 'Graphes', 'phase2': 'Audit CV',
      'phase3': 'Notation', 'reports': 'Rapports', 'bias': 'Audit biais', 'results': 'Résultats'
    };
    return m[c] || c;
  }

  getScoredNodesList(): any[] {
    if (!this.phase3Data?.scored_nodes) return [];
    const nodes = this.phase3Data.scored_nodes;
    if (Array.isArray(nodes)) return nodes;
    return Object.entries(nodes).map(([id, data]: [string, any]) => ({ id, ...data }));
  }

  formatFileSize(octets: number): string {
    if (octets < 1024) return octets + ' o';
    if (octets < 1048576) return (octets / 1024).toFixed(1) + ' Ko';
    return (octets / 1048576).toFixed(1) + ' Mo';
  }

  objectKeys(obj: any): string[] { return obj ? Object.keys(obj) : []; }

  getLiveReportKeys(): number[] {
    return Object.keys(this.liveReports).map(k => +k).sort();
  }

  reset() {
    this._cancelRetry();
    if (this._stopPolling) { this._stopPolling(); this._stopPolling = null; }
    this.state = 'idle';
    this.jdText = '';
    this.selectedFiles = [];
    this.rankings = [];
    this.jobId = '';
    this.errorMsg = '';
    this.charCount = 0;
    this.summaryMsg = '';
    this.graphData = null;
    this.phase2Data = null;
    this.phase3Data = null;
    this.reportData = null;
    this.outputFiles = [];
    this.activeTab = 'overview';
    this.selectedCvIndex = 0;
    this.liveCvs = [];
    this.liveReports = {};
    this._tentativesResultats = 0;
  }
}