from solver import FiniteVol

s = FiniteVol(meshname='base.gri', fluxname='roe', CFL=0.1)

# 1) Check normals orientation
# s.plot_edge_normals(edge_set='both', scale=0.35, stride=3)
s.plot_edge_normals(edge_set='interior', stride=1, scale=0.18, figsize=(14,5))
s.plot_edge_normals(edge_set='boundary', stride=1, scale=0.25, figsize=(14,5))

# 2) Plot residual field at initial state
s.plot_residual_field(s.U0, component='l1', log10=True)

s.plot_normals_with_mesh_labels(
    edge_stride=1,
    show_node_ids=True,
    show_elem_ids=True,
    node_id_stride=1,
    elem_id_stride=1
)


# 3) March some iterations, then plot residual field again
s.solve_steady(runtime=False, itercap=1e2)
s.plot_residual_field(s.U, component='l1', log10=True)

