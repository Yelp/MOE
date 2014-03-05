import colander

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

class ListOfPointsInDomain(colander.SequenceSchema):
    point_in_domain = ListOfFloats()

class GpMeanVarRequest(colander.MappingSchema):
    points_to_sample = ListOfPointsInDomain()
    gp_info = GpInfo()

class GpEiRequest(colander.MappingSchema):
    points_to_evaluate = ListOfPointsInDomain()
    points_being_sampled = ListOfPointsInDomain()
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()

class ListOfExpectedImprovements(colander.SequenceSchema):
    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0),
            )

class GpEiResponse(colander.MappingSchema):
    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = ListOfExpectedImprovements()

class MatrixOfFloats(colander.SequenceSchema):
    row_of_matrix = ListOfFloats()

class GpMeanVarResponse(colander.MappingSchema):
    endpoint = colander.SchemaNode(colander.String())
    mean = ListOfFloats()
    var = MatrixOfFloats()
