from urllib.parse import parse_qs
from fastapi import APIRouter, Request, Response
from .example_loader import load_example

route = APIRouter()


@route.get("/DocumentReference/{id}")
def get_document_reference_by_id(request: Request):
    return Response(status_code=404)


@route.get("/DocumentReference")
def get_document_reference(request: Request):
    q_params = parse_qs(request.url.query)
    subject_ids = q_params.get("subject:identifier")
    if not subject_ids or len(subject_ids) == 0:
        return load_example("bad-request.json")
    if subject_ids[0] == "https://fhir.nhs.uk/Id/nhs-number|4409815415":
        return load_example("document_reference/GET-success-empty.json")
    else:
        return load_example("document_reference/GET-success.json")


@route.post("/DocumentReference")
def post_document_reference(request: Request):
    return Response(status_code=201)


@route.put("/DocumentReference/{id}")
def put_document_reference_by_id(request: Request):
    return Response(status_code=404)


@route.delete("/DocumentReference/{id}")
def delete_document_reference_by_id(request: Request):
    return Response(status_code=200)
