import colander

from optimal_learning.EPI.src.python.constant import default_ei_optimization_parameters, default_gaussian_process_parameters, default_expected_improvement_parameters

class ListOfFloats(colander.SequenceSchema):
    float_in_list = colander.SchemaNode(colander.Float())

class SinglePoint(colander.MappingSchema):
    point = ListOfFloats()
    value = colander.SchemaNode(colander.Float())
    value_var = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0),
            )

class PointsSampled(colander.SequenceSchema):
    point_sampled = SinglePoint()

class DomainCoordinate(colander.SequenceSchema):
    domain_coordinate = colander.SchemaNode(colander.Float())

class Domain(colander.SequenceSchema):
    domain_coordinates = DomainCoordinate()

class GpInfo(colander.MappingSchema):
    points_sampled = PointsSampled()
    domain = Domain()
    length_scale = ListOfFloats(
            missing=default_gaussian_process_parameters.length_scale,
            )
    signal_variance = colander.SchemaNode(
            colander.Float(),
            missing=default_gaussian_process_parameters.signal_variance,
            )

class EiOptimizationParameters(colander.MappingSchema):
    num_multistarts = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.num_multistarts,
            validator=colander.Range(min=1),
            )
    gd_iterations = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.gd_iterations,
            validator=colander.Range(min=10),
            )
    max_num_restarts = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.max_num_restarts,
            validator=colander.Range(min=1),
            )
    gamma = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.gamma,
            validator=colander.Range(min=0.0, max=1.0),
            )
    pre_mult = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.pre_mult,
            validator=colander.Range(min=0.0),
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            missing=default_ei_optimization_parameters.mc_iterations,
            validator=colander.Range(min=100),
            )
    max_relative_change = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.max_relative_change,
            validator=colander.Range(min=0.0, max=1.0),
            )
    tolerance = colander.SchemaNode(
            colander.Float(),
            missing=default_ei_optimization_parameters.tolerance,
            validator=colander.Range(min=1.0e-15, max=1.0e-4),
            )

class ListOfPointsInDomain(colander.SequenceSchema):
    point_in_domain = ListOfFloats()

class GpMeanVarRequest(colander.MappingSchema):
    points_to_sample = ListOfPointsInDomain()
    gp_info = GpInfo()

class GpNextPointsEpiRequest(colander.MappingSchema):
    num_samples_to_generate = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()
    ei_optimization_parameters = EiOptimizationParameters(
            missing=default_ei_optimization_parameters._asdict(),
            )

class GpEiRequest(colander.MappingSchema):
    points_to_evaluate = ListOfPointsInDomain()
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=default_expected_improvement_parameters.mc_iterations,
            )
    gp_info = GpInfo()

class ListOfExpectedImprovements(colander.SequenceSchema):
    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0),
            )

class GpNextPointsEpiResponse(colander.MappingSchema):
    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = ListOfExpectedImprovements()

class GpEiResponse(colander.MappingSchema):
    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = ListOfExpectedImprovements()

class MatrixOfFloats(colander.SequenceSchema):
    row_of_matrix = ListOfFloats()

class GpMeanVarResponse(colander.MappingSchema):
    endpoint = colander.SchemaNode(colander.String())
    mean = ListOfFloats()
    var = MatrixOfFloats()
