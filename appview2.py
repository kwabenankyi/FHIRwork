import json
class CreateAppointment:#for creation only
    def __init__(self, patientID:str, start:str, end:str):
        self.__start = start
        self.__end = end
        self.__participants = []
        self.__finalMessage = {}
        self.setParticipants(patientID)

    def newParticipant(self, participantID:str):
        return {
            "type": [
                {
                    "coding": [
                        {
                            "system": "http://hl7.org/fhir/v3/ParticipationType",
                            "code": "IND"
                        }
                    ]
                }
            ],
            "actor": {
                "reference": participantID
            }
        }
    
    def setParticipants(self, participantID:str):
        self.__participants.append(self.newParticipant(participantID))

    def setFinalMessage(self):
        self.__finalMessage = {
            "resourceType": "Appointment",
            "status": "proposed",
            "start": self.__start,
            "end": self.__end,
            "participant": self.__participants
        }
    def getFinalMessage(self):
        return json.dumps(self.__finalMessage)