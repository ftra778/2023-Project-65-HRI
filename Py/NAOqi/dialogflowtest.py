import dialogflow_v2 as dialogflow



def detect_intent_texts(project_id, session_id, texts, language_code):

    session_client = dialogflow.SessionsClient()

    session = session_client.session_path(project_id, session_id)
    response_text = ""
    print('Session path: {}\n'.format(session))

    for text in texts:
        text_input = dialogflow.types.TextInput(
            text=text, language_code=language_code)

        query_input = dialogflow.types.QueryInput(text=text_input)

        response = session_client.detect_intent(
            session=session, query_input=query_input)

        print('=' * 20)
        for i in response.query_result.fulfillment_messages:
            response_text = response_text + (str(i.text)[7:-2])

        print('Query text: {}'.format(response.query_result.query_text))
        print('Detected intent: {} (confidence: {})\n'.format(
            response.query_result.intent.display_name,
            response.query_result.intent_detection_confidence))
        # print('Fulfillment text: {}\n'.format(
        #    response.query_result.fulfillment_text))
        print("Response text: {}\n".format(response_text))


project_id = "pepperrobot-ppgw"
session_id = "1234567890"
texts = [""]
language_code = "en-US"

while(True):
    texts = [""]
    texts[0] = raw_input("Talk:")
    detect_intent_texts(project_id, session_id, texts, language_code)