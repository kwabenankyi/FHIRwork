from fhirclient import client
import fhirclient.models.patient as p

settings = {
    'app_id': 'my_web_app',
    'api_base': 'https://r3.smarthealthit.org'
}
smart = client.FHIRClient(settings=settings)

patient = p.Patient.read("291112", smart.server)
print(patient.as_json())