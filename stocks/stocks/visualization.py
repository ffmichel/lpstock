import pandas as pd
import networkx as nx

import bokeh.models as bkm
import bokeh.transform as bkt

NODE_JS_CODE = """
    var result = new Float64Array(xs.length)
    for (var i = 0; i < xs.length; i++) {
        result[i] = provider.graph_layout[xs[i]][%s]
    }
    return result
"""

EDGE_JS_CODE = """
    var result = new Float64Array(xs.length)
    coords = provider.get_edge_coordinates(source)[%s]
    for (var i = 0; i < xs.length; i++) {
        result[i] = (coords[i][0] + coords[i][1])/2
    }
    return result
"""


def node_labels(graph_render):
    source = graph_render.node_renderer.data_source

    xcoord = bkm.CustomJSTransform(
        v_func=NODE_JS_CODE % "0",
        args=dict(provider=graph_render.layout_provider))
    ycoord = bkm.CustomJSTransform(
        v_func=NODE_JS_CODE % "1",
        args=dict(provider=graph_render.layout_provider))

    # Use the transforms to supply coords to a LabelSet
    labels = bkm.LabelSet(x=bkt.transform('index', xcoord),
                          y=bkt.transform('index', ycoord), text='index',
                          text_font_size="10px", x_offset=0, y_offset=-5,
                          source=source, render_mode='canvas',
                          text_align='center')
    return labels


def edge_labels(graph_render):
    source = graph_render.edge_renderer.data_source
    xcoord = bkm.CustomJSTransform(
        v_func=EDGE_JS_CODE % "0",
        args=dict(provider=graph_render.layout_provider, source=source))
    ycoord = bkm.CustomJSTransform(
        v_func=EDGE_JS_CODE % "1",
        args=dict(provider=graph_render.layout_provider, source=source))

    source.data['labels'] = [
        str(datum) if not pd.isnull(datum) else ''
        for datum in source.data['relative_allocation']
    ]
    # Use the transforms to supply coords to a LabelSet
    labels = bkm.LabelSet(x=bkt.transform('start', xcoord),
                          y=bkt.transform('start', ycoord), text='labels',
                          text_font_size="12px", x_offset=0, y_offset=0,
                          source=source, render_mode='canvas')
    return labels


def graph_renderer(nx_graph):
    graph_render = bkm.graphs.from_networkx(
        nx_graph,
        nx.drawing.nx_agraph.graphviz_layout,
        prog='dot',
        args='-Nwidth=3 -Gnodesep=1')
    graph_render.node_renderer.glyph = bkm.Ellipse(
        width=max(map(len, nx_graph)) * 2 * 12,
        height=12,
        fill_color="#cab2d6")
    return graph_render
