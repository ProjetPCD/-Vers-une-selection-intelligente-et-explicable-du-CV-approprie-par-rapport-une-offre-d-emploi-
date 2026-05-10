import {
    Component,
    Input,
    OnChanges,
    SimpleChanges,
    ElementRef,
    ViewChild,
    AfterViewInit,
    OnDestroy
} from '@angular/core';
import { CommonModule } from '@angular/common';

export interface GraphNode {
    id: string;
    label: string;
    type: string;
    w_0?: number;
    weight?: number;
    seniority_required?: string;
    context?: string;
    inferred?: boolean;
}

export interface GraphEdge {
    source: string;
    target: string;
    relation?: string;
    weight?: number;
}

export interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    lang?: string;
}

@Component({
    selector: 'app-graph-viewer',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './graph-viewer.html',
    styleUrl: './graph-viewer.scss'
})
export class GraphViewerComponent implements OnChanges, AfterViewInit, OnDestroy {
    @Input() graphData: GraphData | null = null;
    @Input() title = 'Knowledge Graph';

    @ViewChild('svgContainer') svgContainerRef!: ElementRef<HTMLDivElement>;

    private simulation: any = null;
    private svg: any = null;
    private d3Loaded = false;
    private resizeObserver: ResizeObserver | null = null;

    readonly TYPE_COLORS: Record<string, string> = {
        'Hard Skill': '#1f77b4',
        'Soft Skill': '#ff7f0e',
        'Responsibility': '#2ca02c',
        'Tool': '#d62728',
        'Domain': '#9467bd',
        'Experience': '#e377c2',
        'Certification': '#bcbd22',
        'Education': '#500911',
        'Unknown': '#A9A9A9'
    };

    get nodeTypes(): string[] {
        if (!this.graphData?.nodes) return [];
        return [...new Set(this.graphData.nodes.map(n => n.type || 'Unknown'))];
    }

    get totalEdges(): number {
        return this.graphData?.edges?.length ?? 0;
    }

    get totalNodes(): number {
        return this.graphData?.nodes?.length ?? 0;
    }

    typeColor(type: string): string {
        return this.TYPE_COLORS[type] ?? this.TYPE_COLORS['Unknown'];
    }

    countByType(type: string): number {
        return this.graphData?.nodes?.filter(n => (n.type || 'Unknown') === type).length ?? 0;
    }

    async ngAfterViewInit() {
        await this.loadD3();
        this.renderGraph();
        this.setupResizeObserver();
    }

    ngOnChanges(changes: SimpleChanges) {
        if (changes['graphData'] && this.d3Loaded && this.svgContainerRef) {
            this.renderGraph();
        }
    }

    ngOnDestroy() {
        this.resizeObserver?.disconnect();
        if (this.simulation) this.simulation.stop();
    }

    private loadD3(): Promise<void> {
        return new Promise((resolve) => {
            if ((window as any).d3) {
                this.d3Loaded = true;
                resolve();
                return;
            }
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js';
            script.onload = () => {
                this.d3Loaded = true;
                resolve();
            };
            document.head.appendChild(script);
        });
    }

    private setupResizeObserver() {
        if (!this.svgContainerRef?.nativeElement) return;
        this.resizeObserver = new ResizeObserver(() => this.renderGraph());
        this.resizeObserver.observe(this.svgContainerRef.nativeElement);
    }

    renderGraph() {
        if (!this.d3Loaded || !this.svgContainerRef || !this.graphData) return;

        const d3 = (window as any).d3;
        const container = this.svgContainerRef.nativeElement;
        const W = container.clientWidth || 800;
        const H = container.clientHeight || 500;

        // Clear previous
        d3.select(container).selectAll('*').remove();
        if (this.simulation) this.simulation.stop();

        if (!this.graphData.nodes?.length) return;

        const nodes: any[] = this.graphData.nodes.map(n => ({ ...n }));
        const nodeById = new Map(nodes.map(n => [n.id, n]));
        const edges: any[] = (this.graphData.edges || [])
            .filter(e => nodeById.has(e.source) && nodeById.has(e.target))
            .map(e => ({ ...e }));

        const svg = d3.select(container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${W} ${H}`)
            .style('overflow', 'visible');

        // Defs: arrowhead + glow filter
        const defs = svg.append('defs');
        defs.append('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 22)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', 'rgba(255,255,255,0.25)');

        const filter = defs.append('filter').attr('id', 'glow');
        filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'blur');
        const feMerge = filter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'blur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Zoom
        const g = svg.append('g');
        svg.call(d3.zoom().scaleExtent([0.3, 3])
            .on('zoom', (event: any) => g.attr('transform', event.transform)));

        // Links
        const link = g.append('g')
            .selectAll('line')
            .data(edges)
            .join('line')
            .attr('stroke', 'rgba(255,255,255,0.15)')
            .attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrow)');

        // Link labels
        const linkLabel = g.append('g')
            .selectAll('text')
            .data(edges.filter((e: any) => e.relation))
            .join('text')
            .attr('font-size', '9px')
            .attr('fill', 'rgba(255,255,255,0.35)')
            .attr('text-anchor', 'middle')
            .text((d: any) => d.relation || '');

        // Node groups
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .join('g')
            .attr('cursor', 'pointer')
            .call(d3.drag()
                .on('start', (event: any, d: any) => {
                    if (!event.active) this.simulation.alphaTarget(0.3).restart();
                    d.fx = d.x; d.fy = d.y;
                })
                .on('drag', (event: any, d: any) => { d.fx = event.x; d.fy = event.y; })
                .on('end', (event: any, d: any) => {
                    if (!event.active) this.simulation.alphaTarget(0);
                    d.fx = null; d.fy = null;
                })
            );

        // Circle + glow
        node.append('circle')
            .attr('r', (d: any) => 10 + (d.w_0 || d.weight || 0.5) * 18)
            .attr('fill', (d: any) => this.typeColor(d.type || 'Unknown'))
            .attr('fill-opacity', 0.85)
            .attr('stroke', (d: any) => this.typeColor(d.type || 'Unknown'))
            .attr('stroke-width', 2)
            .attr('filter', 'url(#glow)');

        // Labels
        node.append('text')
            .attr('dy', '0.35em')
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('font-weight', '600')
            .attr('fill', '#fff')
            .attr('pointer-events', 'none')
            .text((d: any) => {
                const lbl = d.label || d.id || '';
                return lbl.length > 16 ? lbl.slice(0, 14) + '…' : lbl;
            });

        // Tooltip
        const tooltip = d3.select(container)
            .append('div')
            .attr('class', 'graph-tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(15,15,25,0.95)')
            .style('border', `1px solid rgba(255,255,255,0.15)`)
            .style('border-radius', '8px')
            .style('padding', '8px 12px')
            .style('font-size', '12px')
            .style('color', '#fff')
            .style('pointer-events', 'none')
            .style('opacity', 0)
            .style('max-width', '200px')
            .style('z-index', '100');

        node.on('mouseover', (event: any, d: any) => {
            const color = this.typeColor(d.type || 'Unknown');
            tooltip.html(`
        <div style="color:${color};font-weight:700;margin-bottom:4px">${d.label || d.id}</div>
        <div style="opacity:0.7">${d.type || 'Unknown'}</div>
        ${d.weight != null ? `<div style="opacity:0.5;font-size:11px">weight: ${d.weight}</div>` : ''}
      `)
                .style('opacity', 1)
                .style('left', (event.offsetX + 12) + 'px')
                .style('top', (event.offsetY - 10) + 'px');
        }).on('mousemove', (event: any) => {
            tooltip.style('left', (event.offsetX + 12) + 'px')
                .style('top', (event.offsetY - 10) + 'px');
        }).on('mouseout', () => tooltip.style('opacity', 0));

        // Simulation
        this.simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id((d: any) => d.id).distance(100).strength(0.5))
            .force('charge', d3.forceManyBody().strength(-250))
            .force('center', d3.forceCenter(W / 2, H / 2))
            .force('collision', d3.forceCollide(30))
            .on('tick', () => {
                link
                    .attr('x1', (d: any) => d.source.x)
                    .attr('y1', (d: any) => d.source.y)
                    .attr('x2', (d: any) => d.target.x)
                    .attr('y2', (d: any) => d.target.y);

                linkLabel
                    .attr('x', (d: any) => (d.source.x + d.target.x) / 2)
                    .attr('y', (d: any) => (d.source.y + d.target.y) / 2);

                node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
            });
    }
}