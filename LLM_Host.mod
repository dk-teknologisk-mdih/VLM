MODULE LLM_Host

    !*****************************************************
    ! LLM Code Execution Host Program
    ! 
    ! This module acts as a TCP server that listens for
    ! commands from a Python orchestration script.
    ! It can dynamically load, execute, and unload
    ! LLM-generated RAPID modules.
    !
    ! Protocol (text-based over TCP):
    !   Python -> Robot:
    !     "LOAD:<filename>"    e.g. "LOAD:generated_task.mod"
    !     "START"              execute the loaded module
    !     "PING"               health check
    !     "QUIT"               close connection gracefully
    !
    !   Robot -> Python:
    !     "READY"              sent on initial connection
    !     "LOAD_OK"            module loaded successfully
    !     "LOAD_ERR:<reason>"  module load failed
    !     "START_OK"           execution started
    !     "DONE"               execution finished successfully
    !     "EXEC_ERR:<reason>"  execution failed
    !     "PONG"               health check response
    !     "BYE"                connection closing
    !     "UNKNOWN_CMD"        unrecognized command
    !*****************************************************

    ! ---- Configuration constants ----
    ! Change ROBOT_IP to match your controller's LAN IP
    CONST string ROBOT_IP := "192.168.125.1";
    CONST num SERVER_PORT := 1025;
    CONST string MODULE_DIR := "FTP_RobotPrograms/";

    ! ---- Socket variables ----
    VAR socketdev server_socket;
    VAR socketdev client_socket;
    VAR string client_ip;

    ! ---- State variables ----
    VAR string received_msg;
    VAR string loaded_file_path;
    VAR bool module_is_loaded := FALSE;
    VAR bool client_connected := FALSE;

    !*****************************************************
    ! MAIN - Entry point
    !*****************************************************
    PROC main()
        ! Deactivate configuration supervision for
        ! compatibility with dynamically generated targets
        ConfJ \Off;
        ConfL \Off;

        TPWrite "LLM Host: Starting up...";

        ! Main outer loop: reconnect if client disconnects
        WHILE TRUE DO
            setup_server;
            wait_for_client;
            command_loop;
            cleanup_client;
        ENDWHILE
    ENDPROC

    !*****************************************************
    ! SETUP_SERVER - Create and bind the server socket
    !*****************************************************
    PROC setup_server()
        ! Close any leftover sockets (safe even if not created)
        SocketClose server_socket;
        SocketClose client_socket;

        SocketCreate server_socket;
        SocketBind server_socket, ROBOT_IP, SERVER_PORT;
        SocketListen server_socket;
        TPWrite "LLM Host: Listening on " + ROBOT_IP + ":" + NumToStr(SERVER_PORT, 0);

    ERROR
        IF ERRNO = ERR_SOCK_ADDR_INUSE THEN
            TPWrite "LLM Host: Port in use, retrying in 5s...";
            WaitTime 5;
            RETRY;
        ELSEIF ERRNO = ERR_SOCK_CLOSED THEN
            RETRY;
        ENDIF
    ENDPROC

    !*****************************************************
    ! WAIT_FOR_CLIENT - Accept incoming Python connection
    !*****************************************************
    PROC wait_for_client()
        TPWrite "LLM Host: Waiting for Python client...";
        SocketAccept server_socket, client_socket \ClientAddress:=client_ip \Time:=WAIT_MAX;
        client_connected := TRUE;
        TPWrite "LLM Host: Client connected from " + client_ip;

        ! Send READY handshake
        SocketSend client_socket \Str:="READY\0A";

    ERROR
        IF ERRNO = ERR_SOCK_TIMEOUT THEN
            RETRY;
        ELSEIF ERRNO = ERR_SOCK_CLOSED THEN
            TPWrite "LLM Host: Socket closed during accept, restarting...";
            RAISE;
        ENDIF
    ENDPROC

    !*****************************************************
    ! COMMAND_LOOP - Main loop processing client commands
    !*****************************************************
    PROC command_loop()
        VAR string cmd;
        VAR string arg;
        VAR num sep_pos;

        WHILE client_connected DO
            ! Wait for a command (block up to 300s, then check)
            received_msg := "";
            SocketReceive client_socket \Str:=received_msg \Time:=300;

            ! Strip any trailing whitespace / newline chars
            received_msg := strip_newline(received_msg);

            ! Parse command and argument (split on ":")
            sep_pos := str_find(received_msg, ":");
            IF sep_pos > 0 THEN
                cmd := StrPart(received_msg, 1, sep_pos - 1);
                arg := StrPart(received_msg, sep_pos + 1, StrLen(received_msg) - sep_pos);
            ELSE
                cmd := received_msg;
                arg := "";
            ENDIF

            ! Dispatch command
            TEST cmd
            CASE "LOAD":
                handle_load arg;
            CASE "START":
                handle_start;
            CASE "PING":
                send_response "PONG";
            CASE "QUIT":
                send_response "BYE";
                client_connected := FALSE;
            DEFAULT:
                send_response "UNKNOWN_CMD";
            ENDTEST
        ENDWHILE

    ERROR
        IF ERRNO = ERR_SOCK_TIMEOUT THEN
            ! No command received within timeout, just keep waiting
            RETRY;
        ELSEIF ERRNO = ERR_SOCK_CLOSED THEN
            TPWrite "LLM Host: Client disconnected.";
            client_connected := FALSE;
            TRYNEXT;
        ENDIF
    ENDPROC

    !*****************************************************
    ! HANDLE_LOAD - Load a dynamically generated module
    !*****************************************************
    PROC handle_load(string filename)
        VAR string full_path;

        ! If a module is already loaded, unload it first
        IF module_is_loaded THEN
            unload_current_module;
        ENDIF

        full_path := MODULE_DIR + filename;
        loaded_file_path := full_path;

        TPWrite "LLM Host: Loading " + full_path;
        Load \Dynamic, full_path \CheckRef;
        module_is_loaded := TRUE;
        TPWrite "LLM Host: Load OK";
        send_response "LOAD_OK";

    ERROR
        IF ERRNO = ERR_FILNOTFND THEN
            TPWrite "LLM Host: File not found: " + loaded_file_path;
            send_response "LOAD_ERR:FILE_NOT_FOUND";
            module_is_loaded := FALSE;
            RETURN;
        ELSEIF ERRNO = ERR_SYNTAX THEN
            TPWrite "LLM Host: Syntax error in module";
            send_response "LOAD_ERR:SYNTAX_ERROR";
            module_is_loaded := FALSE;
            RETURN;
        ELSEIF ERRNO = ERR_LINKREF THEN
            TPWrite "LLM Host: Unresolved references in module";
            send_response "LOAD_ERR:LINK_ERROR";
            module_is_loaded := FALSE;
            RETURN;
        ELSEIF ERRNO = ERR_LOADED THEN
            TPWrite "LLM Host: Module already loaded, unloading first...";
            UnLoad loaded_file_path;
            module_is_loaded := FALSE;
            RETRY;
        ELSEIF ERRNO = ERR_PRGMEMFULL THEN
            TPWrite "LLM Host: Program memory full";
            send_response "LOAD_ERR:MEMORY_FULL";
            module_is_loaded := FALSE;
            RETURN;
        ELSEIF ERRNO = ERR_IOERROR THEN
            TPWrite "LLM Host: I/O error reading file";
            send_response "LOAD_ERR:IO_ERROR";
            module_is_loaded := FALSE;
            RETURN;
        ENDIF
    ENDPROC

    !*****************************************************
    ! HANDLE_START - Execute the loaded module's run_task
    !                procedure via late binding, then unload
    !*****************************************************
    PROC handle_start()
        IF NOT module_is_loaded THEN
            send_response "EXEC_ERR:NO_MODULE_LOADED";
            RETURN;
        ENDIF

        TPWrite "LLM Host: Executing run_task...";
        send_response "START_OK";

        ! Call the entry-point procedure in the loaded module
        ! using late binding (% "name" % syntax).
        ! The LLM-generated module MUST define a procedure
        ! called "run_task" as its entry point.
        %"run_task"%;

        TPWrite "LLM Host: Execution complete.";

        ! Unload the module now that execution has returned
        unload_current_module;

        send_response "DONE";

    ERROR
        IF ERRNO = ERR_REFUNKPRC THEN
            TPWrite "LLM Host: run_task procedure not found in module";
            SkipWarn;
            send_response "EXEC_ERR:PROC_NOT_FOUND";
            ! Try to unload the broken module
            unload_current_module;
            TRYNEXT;
        ELSE
            TPWrite "LLM Host: Runtime error during execution, ERRNO=" + NumToStr(ERRNO, 0);
            send_response "EXEC_ERR:RUNTIME_" + NumToStr(ERRNO, 0);
            ! Try to unload
            unload_current_module;
            TRYNEXT;
        ENDIF
    ENDPROC

    !*****************************************************
    ! UNLOAD_CURRENT_MODULE - Safely unload the loaded module
    !*****************************************************
    PROC unload_current_module()
        IF module_is_loaded THEN
            TPWrite "LLM Host: Unloading " + loaded_file_path;
            UnLoad loaded_file_path;
            module_is_loaded := FALSE;
            TPWrite "LLM Host: Unload OK";
        ENDIF

    ERROR
        IF ERRNO = ERR_UNLOAD THEN
            TPWrite "LLM Host: Warning - could not unload module";
            module_is_loaded := FALSE;
            TRYNEXT;
        ENDIF
    ENDPROC

    !*****************************************************
    ! CLEANUP_CLIENT - Close client socket after disconnect
    !*****************************************************
    PROC cleanup_client()
        ! Unload any loaded module before accepting new client
        unload_current_module;

        WaitTime 2;
        SocketClose client_socket;
        client_connected := FALSE;
        TPWrite "LLM Host: Client session ended.";

    ERROR
        TRYNEXT;
    ENDPROC

    !*****************************************************
    ! SEND_RESPONSE - Send a string response to the Python
    !                 client, appending a newline delimiter
    !*****************************************************
    PROC send_response(string msg)
        SocketSend client_socket \Str:=msg + "\0A";

    ERROR
        IF ERRNO = ERR_SOCK_CLOSED THEN
            TPWrite "LLM Host: Cannot send, client disconnected";
            client_connected := FALSE;
            TRYNEXT;
        ENDIF
    ENDPROC

    !*****************************************************
    ! HELPER: str_find - Find first occurrence of a 
    !         character in a string. Returns 0 if not found.
    !*****************************************************
    FUNC num str_find(string source, string char)
        VAR num i;
        FOR i FROM 1 TO StrLen(source) DO
            IF StrPart(source, i, 1) = char THEN
                RETURN i;
            ENDIF
        ENDFOR
        RETURN 0;
    ENDFUNC

    !*****************************************************
    ! HELPER: strip_newline - Remove trailing \0A \0D chars
    !*****************************************************
    FUNC string strip_newline(string s)
        VAR num len;
        len := StrLen(s);
        WHILE len > 0 DO
            IF StrPart(s, len, 1) = "\0A" OR StrPart(s, len, 1) = "\0D" THEN
                len := len - 1;
            ELSE
                RETURN StrPart(s, 1, len);
            ENDIF
        ENDWHILE
        RETURN "";
    ENDFUNC

ENDMODULE