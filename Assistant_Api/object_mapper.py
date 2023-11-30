import json




def serialize_assistant(my_assistant):
    # Map Assistant object attributes to a dictionary for serialization
    assistant_info = {
        "id": my_assistant.id,
        "object": my_assistant.object, 
        "created_at": my_assistant.created_at,
        "name": my_assistant.name,
        "description": my_assistant.description,
        "model": my_assistant.model
    }
    return json.dumps(assistant_info, indent=2)




def serialize_chat_thread(chat_thread):
    # Map ChatThread object attributes to a dictionary for serialization
    chat_thread_info = {
        "id": chat_thread.id,
        "object": chat_thread.object,
        "created_at": chat_thread.created_at
    }
    return json.dumps(chat_thread_info, indent=2)





def serialize_thread_message(thread_messages):
    serialized_messages = []
    # print(thread_messages)

    for thread_message in thread_messages:
        # print(thread_message)
        serialized_message = {
            "id": thread_message.id,
            "object": thread_message.object,
            "created_at": thread_message.created_at,
            "thread_id": thread_message.thread_id,
            "role": thread_message.role,
            "message": thread_message.content[0].text.value,
            "assistant_id": thread_message.assistant_id,
            "run_id": thread_message.run_id
        }
        serialized_messages.append(serialized_message)
    
    return json.dumps(serialized_messages, indent=2)








def serialize_run(run):
    # Map Run object attributes to a dictionary for serialization
    run_info = {
        "id": run.id,
        "object": run.object,
        "created_at": run.created_at,
        "thread_id": run.thread_id,
        "assistant_id": run.assistant_id,
        "status": run.status,
        "expires_at": run.expires_at
    }
    return json.dumps(run_info, indent=2)