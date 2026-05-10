import { Component } from '@angular/core';
import { MatchFormComponent } from './components/match-form/match-form';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [MatchFormComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  title = 'CV Matcher';
}
