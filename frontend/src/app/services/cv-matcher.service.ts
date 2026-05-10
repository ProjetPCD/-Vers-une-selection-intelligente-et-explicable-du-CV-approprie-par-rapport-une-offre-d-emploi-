import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface SubmitResponse {
  job_id: string;
  message: string;
}

export interface StatusResponse {
  job_id: string;
  status: string;
  current_phase?: string;
  progress?: string;
  summary?: string;
}

export interface CVRankItem {
  cv_index: number;
  pair_key: string;
  rank: number | null;
  tier: string | null;
  verdict: string | null;
  match_pct: number | null;
  combo_v6: number | null;
  confidence: string | null;
}

export interface ResultsResponse {
  job_id: string;
  rankings: CVRankItem[];
}

export interface OutputFile {
  name: string;
  category: string;
  size: number;
  path: string;
}

export interface OutputsResponse {
  job_id: string;
  files: OutputFile[];
}

export interface AvailableCv {
  cv_index: number;
  has_phase2: boolean;
  has_phase3: boolean;
  has_report: boolean;
}

export interface AvailableResponse {
  graph: boolean;
  graph_on_disk: boolean;
  cvs: AvailableCv[];
}

@Injectable({ providedIn: 'root' })
export class CvMatcherService {

  private api = 'http://localhost:8000/api';

  constructor(private http: HttpClient) { }

  submitMatch(jobDescription: string, cvFiles: File[]): Observable<SubmitResponse> {
    const form = new FormData();
    form.append('job_description', jobDescription);
    cvFiles.forEach(f => form.append('cvs', f, f.name));
    return this.http.post<SubmitResponse>(`${this.api}/match`, form);
  }

  getStatus(jobId: string): Observable<StatusResponse> {
    return this.http.get<StatusResponse>(`${this.api}/status/${jobId}`);
  }

  getResults(jobId: string): Observable<ResultsResponse> {
    return this.http.get<ResultsResponse>(`${this.api}/results/${jobId}`);
  }

  getGraph(jobId: string): Observable<any> {
    return this.http.get<any>(`${this.api}/graph/${jobId}`);
  }

  getPhase2(jobId: string, cvIndex: number): Observable<any> {
    return this.http.get<any>(`${this.api}/phase2/${jobId}/${cvIndex}`);
  }

  getPhase3(jobId: string, cvIndex: number): Observable<any> {
    return this.http.get<any>(`${this.api}/phase3/${jobId}/${cvIndex}`);
  }

  getReport(jobId: string, cvIndex: number): Observable<any> {
    return this.http.get<any>(`${this.api}/report/${jobId}/${cvIndex}`);
  }

  getOutputs(jobId: string): Observable<OutputsResponse> {
    return this.http.get<OutputsResponse>(`${this.api}/outputs/${jobId}`);
  }

  getOutputFileUrl(path: string): string {
    return `http://localhost:8000${path}`;
  }

  getAvailable(jobId: string): Observable<AvailableResponse> {
    return this.http.get<AvailableResponse>(`${this.api}/available/${jobId}`);
  }

  /**
   * Polling robuste basé sur setTimeout récursif.
   * Aucune race condition possible : une seule requête en vol à la fois.
   * Retourne une fonction stop() pour annuler le polling.
   */
  startPollingLoop(
    jobId: string,
    onStatus: (res: StatusResponse) => void,
    onDone: (res: StatusResponse) => void,
    onFatalError: () => void
  ): () => void {
    let stopped = false;
    let timerId: any = null;
    let consecutiveErrors = 0;
    const MAX_ERRORS = 10;

    const tick = () => {
      if (stopped) return;
      this.getStatus(jobId).subscribe({
        next: (res: StatusResponse) => {
          if (stopped) return;
          consecutiveErrors = 0;
          onStatus(res);
          if (res.status === 'done' || res.status === 'error') {
            stopped = true;
            onDone(res);
          } else {
            timerId = setTimeout(tick, 2000);
          }
        },
        error: (err: any) => {
          if (stopped) return;
          consecutiveErrors++;
          console.warn(`[Polling] Erreur réseau (${consecutiveErrors}/${MAX_ERRORS})`, err);
          if (consecutiveErrors >= MAX_ERRORS) {
            stopped = true;
            onFatalError();
          } else {
            timerId = setTimeout(tick, 3000);
          }
        }
      });
    };

    timerId = setTimeout(tick, 1000);

    return () => {
      stopped = true;
      if (timerId !== null) clearTimeout(timerId);
    };
  }
}