"""Drawing the network.

Two views of the graph network.py built, for looking at rather than computing
with. Nothing else in the pipeline depends on these, and skipping them changes
no result.

    draw_colored_network()  a flat 2D diagram, nodes coloured by brain region
                            and shaped by role (target, driver, external leaf).
    plot_3d_network()       an interactive HTML view, written to results_dir and
                            opened in a browser. Layered by depth from the
                            targets, so the backbone Stage 1 built reads as
                            distance from the behaviour being explained.

Depends on: nothing in the package; networkx, matplotlib.
Imported by: pipeline, the entry scripts.
"""

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')          # cluster jobs have no display
import matplotlib.pyplot as plt
import networkx as nx


def draw_colored_network(net_result, target_colors, save_path):
    '''Draws the network graph, nodes colored by which target ultimately
    depends on them. Edges are colored by delay, with a legend mapping
    color -> delay, whenever the graph carries per-edge delay data (a
    MultiDiGraph built by build_nested_mde_network() for MODE='MDE-GMN').
    MODE='GMN' networks (a plain DiGraph from apps.CreateNetwork, no delay
    concept at all) fall back to the original by-target edge coloring.
    Ported unchanged from simple-mde-gmn.py -- purely graph-structural, no
    dependency on what the columns represent.'''
    graph = net_result['network']['Graph']
    targets = net_result['targets']
    has_delays = isinstance(graph, nx.MultiDiGraph)

    node_colors = [target_colors.get(n, 'lightgray') for n in graph.nodes]

    pos = nx.arf_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=300, node_color=node_colors)
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=9, font_weight='bold')

    if has_delays:
        all_delays = sorted({data['delay'] for _, _, data in graph.edges(data=True)})
        cmap = plt.get_cmap('viridis')
        delay_color = {d: cmap(i / max(1, len(all_delays) - 1)) for i, d in enumerate(all_delays)}

        parallel_count = defaultdict(int)
        for u, v, data in graph.edges(data=True):
            i = parallel_count[(u, v)]
            parallel_count[(u, v)] += 1
            rad = 0.15 * ((i + 1) // 2) * (1 if i % 2 == 0 else -1)
            nx.draw_networkx_edges(
                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
                edge_color=[delay_color[data['delay']]],
                connectionstyle=f'arc3,rad={rad}',
            )

        edge_handles = [
            plt.Line2D([0], [0], color=delay_color[d], lw=2, label=f'delay={d}')
            for d in all_delays
        ]
        legend_title = 'Edge delay / target'
    else:
        for u, v in graph.edges():
            nx.draw_networkx_edges(
                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
                edge_color=[target_colors.get(v, 'gray')],
            )
        edge_handles = []
        legend_title = 'Target'

    node_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                   markerfacecolor=target_colors[t], label=t)
        for t in targets
    ]
    ax.legend(handles=edge_handles + node_handles, loc='upper left',
              bbox_to_anchor=(1.0, 1.0), title=legend_title)
    ax.set_title(f"GMN network ({net_result['network_id']})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_3d_network(net_result, target_colors, save_path):
    '''Interactive HTML redesign of the 3D network figure (superseding the
    earlier static-PNG version this docstring used to describe). MODE=
    'MDE-GMN' only (needs the per-edge delay data and interior/external
    partition only that mode's MultiDiGraph has, both absent from MODE=
    'GMN''s plain DiGraph).

    x-y plane (z=0): only the INTERIOR nodes (>=1 discovered driver --
    genuine Stage 1/2 network members, partition_network()'s own
    classification read from graph.graph, not re-derived here), laid out
    via a spring layout of the real-observable (delay=0) edges between two
    interior nodes -- the same structural skeleton draw_colored_network()
    draws in 2D, just restricted to interior nodes since external inputs
    are handled entirely separately below.

    Everything that drives an interior node but has no drivers of its own
    is lifted onto the z-axis as its own distinct "entity" -- never stacked
    directly above whichever node it happens to feed (the old design's
    approach, which duplicated a driver once per consumer). One entity per
    distinct (driver, delay!=0) pair (a lagged copy of an observable -- this
    covers both a genuine external input's lagged edge and an interior
    node's own AR self-loop, since a lagged copy is conceptually distinct
    from that observable's own real-time position either way), labeled
    f"{driver}_delay_{delay}"; plus one entity per external-input driver
    whose edge has delay==0 (a real-observable leaf), labeled with its own
    plain name. Each entity gets a single (x, y) position solved by a second
    spring-layout pass with the interior positions held fixed (see
    _entity_layout()), so a single driver feeding several nodes appears once
    with arrows fanning out to each of them, rather than duplicated per
    consumer. z = |delay| for a lagged-copy entity, or a small fixed
    elevation for a delay==0 leaf entity (enough to read as "off the plane"
    without implying a false delay magnitude).

    Every graph edge becomes one directed arrow (a plotly Cone trace,
    plotly's own 3D substitute for a directed arrowhead -- there is no 3D
    analog of draw_networkx_edges()'s 2D arrows) between its resolved
    driver position and its driven node's plane position, colored by delay
    using the same viridis mapping draw_colored_network() uses so the two
    figures read consistently. Nodes are drawn as three marker shapes --
    circle (interior), diamond (real-observable leaf entity), square
    (lagged-copy entity, colored by delay rather than by target) -- the
    same 3-way distinction the old design drew via circle/triangle/square
    (plotly's 3D markers have no triangle symbol; diamond substitutes).
    Rendered via plotly rather than matplotlib specifically for the
    interactivity (rotate/zoom/hover) a static image can't offer.'''
    import plotly.graph_objects as go
    import matplotlib.colors as mcolors

    def _color(c):
        '''Normalizes any color target_colors/our own fallbacks/viridis's
        cmap() might hand us -- a matplotlib named string ('tab:red'), a raw
        RGBA float tuple (what cfg['TARGET_COLORS'] actually contains in
        production, via plt.get_cmap('tab10')(i % 10)), or an already-hex
        string -- into a plotly-safe '#rrggbb' hex string. Plotly's color
        validators reject bare RGBA tuples (mistaken for numeric colorscale
        data) and non-CSS names like 'tab:red'.'''
        return mcolors.to_hex(mcolors.to_rgba(c))

    graph = net_result['network']['Graph']
    interior_nodes = graph.graph.get('interior_nodes', set())

    all_delays = sorted({d['delay'] for _, _, d in graph.edges(data=True)})
    cmap = plt.get_cmap('viridis')
    delay_color = {
        d: _color(cmap(i / max(1, len(all_delays) - 1)))
        for i, d in enumerate(all_delays)
    }
    nonzero_abs_delays = sorted({abs(d) for d in all_delays if d != 0})
    leaf_z = nonzero_abs_delays[0] * 0.5 if nonzero_abs_delays else 1.0

    # ---- pass 1: interior-only real-observable backbone layout (z=0) ----
    interior_real_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d['delay'] == 0 and u in interior_nodes and v in interior_nodes
    ]
    backbone = nx.DiGraph()
    backbone.add_nodes_from(interior_nodes)
    backbone.add_edges_from(interior_real_edges)
    interior_pos = nx.spring_layout(backbone, seed=0) if interior_nodes else {}

    # ---- classify every edge into "already on the plane" (interior ->
    # interior, delay==0) or "driven by an elevated entity" ----
    entity_meta = {}     # label -> {'driver':.., 'delay':.., 'kind': 'leaf'|'delay_copy'}
    entity_out_edges = []  # (label, driven_node) -- for the pass-2 layout graph
    for u, v, data in graph.edges(data=True):
        delay = data['delay']
        if delay == 0 and u in interior_nodes:
            continue  # part of the pass-1 backbone
        if delay == 0:
            # u is necessarily an external input here -- external inputs
            # have in_degree 0 by partition_network()'s own definition, so
            # they can never be a driven node v, and any interior u with
            # delay==0 was already excluded just above.
            label, kind = u, 'leaf'
        else:
            label, kind = f'{u}_delay_{delay}', 'delay_copy'
        entity_meta[label] = {'driver': u, 'delay': delay, 'kind': kind}
        entity_out_edges.append((label, v))

    # ---- pass 2: constrained layout -- interior positions held fixed,
    # solve (x, y) only for the new elevated entities. spring_layout
    # auto-seeds any node present in `combined` but absent from `pos=` at a
    # random position scaled to the *given* positions' own magnitude, so no
    # manual initial guess is needed for the new entity nodes. ----
    combined = nx.DiGraph()
    combined.add_nodes_from(interior_nodes)
    combined.add_edges_from(entity_out_edges)
    if interior_nodes:
        solved_pos = nx.spring_layout(
            combined, pos=dict(interior_pos), fixed=list(interior_nodes), seed=0,
        )
    else:
        solved_pos = nx.spring_layout(combined, seed=0) if combined.nodes else {}

    pos2d = dict(interior_pos)
    for label in entity_meta:
        pos2d[label] = solved_pos.get(label, (0.0, 0.0))

    def _xyz(name):
        x, y = pos2d[name]
        if name in entity_meta:
            m = entity_meta[name]
            z = abs(m['delay']) if m['kind'] == 'delay_copy' else leaf_z
        else:
            z = 0.0
        return float(x), float(y), float(z)

    def _src_label(u, delay):
        if delay == 0 and u in interior_nodes:
            return u
        if delay == 0:
            return u
        return f'{u}_delay_{delay}'

    # ---- edge traces, grouped by delay (line + arrowhead cone per group) ----
    edges_by_delay = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        delay = data['delay']
        edges_by_delay[delay].append((_xyz(_src_label(u, delay)), _xyz(v)))

    fig = go.Figure()
    for delay, segs in sorted(edges_by_delay.items()):
        color = delay_color[delay]
        xs, ys, zs = [], [], []
        for (x0, y0, z0), (x1, y1, z1) in segs:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=color, width=3),
            name=f'delay={delay}', legendgroup=f'delay_{delay}',
            hoverinfo='none', opacity=0.7,
        ))

        cx, cy, cz, cu, cv, cw = [], [], [], [], [], []
        for (x0, y0, z0), (x1, y1, z1) in segs:
            tx, ty, tz = x1 - x0, y1 - y0, z1 - z0
            cx.append(x0 + 0.88 * tx); cy.append(y0 + 0.88 * ty); cz.append(z0 + 0.88 * tz)
            cu.append(tx); cv.append(ty); cw.append(tz)
        fig.add_trace(go.Cone(
            x=cx, y=cy, z=cz, u=cu, v=cv, w=cw,
            anchor='tail', sizemode='absolute', sizeref=0.15,
            colorscale=[[0, color], [1, color]], showscale=False,
            legendgroup=f'delay_{delay}', showlegend=False, hoverinfo='none',
        ))

    # ---- node traces: interior (circle), leaf (diamond), delay-copy (square) ----
    def _node_trace(names, symbol, colors, size, label):
        if not names:
            return None
        xs, ys, zs = zip(*(_xyz(n) for n in names))
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode='markers+text', text=list(names),
            textposition='top center', textfont=dict(size=9),
            marker=dict(symbol=symbol, size=size, color=list(colors),
                        line=dict(color='black', width=1)),
            name=label,
        )

    interior_list = sorted(interior_nodes)
    leaf_list = sorted(n for n, m in entity_meta.items() if m['kind'] == 'leaf')
    delay_copy_list = sorted(n for n, m in entity_meta.items() if m['kind'] == 'delay_copy')

    for t in (
        _node_trace(interior_list, 'circle',
                    [_color(target_colors.get(n, 'tab:blue')) for n in interior_list],
                    8, f'interior (n={len(interior_list)})'),
        _node_trace(leaf_list, 'diamond',
                    [_color(target_colors.get(entity_meta[n]['driver'], 'tab:orange'))
                     for n in leaf_list],
                    7, f'external input, real-time (n={len(leaf_list)})'),
        _node_trace(delay_copy_list, 'square',
                    [delay_color[entity_meta[n]['delay']] for n in delay_copy_list],
                    6, f'lagged driver copy (n={len(delay_copy_list)})'),
    ):
        if t is not None:
            fig.add_trace(t)

    fig.update_layout(
        scene=dict(
            xaxis_title='X (interior network layout)',
            yaxis_title='Y (interior network layout)',
            zaxis_title='|delay| (0 = real-time leaf elevation)',
        ),
        title=f"3D network projection ({net_result['network_id']})",
        legend_title=f"n_interior={len(interior_list)}, "
                     f"n_external_leaf={len(leaf_list)}, n_lagged_copy={len(delay_copy_list)}",
    )
    fig.write_html(save_path)
