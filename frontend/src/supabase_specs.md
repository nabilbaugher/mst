Main supabase database under this:

Zoe's github (zsheill7)

Main table:

tester_id: uuid
created_at: timestamp
keystroke_sequences: json

Example for keystroke_sequences:

Each person might do up to 10 rounds, so we want to save their data
in 10 different keystroke sequences

{
    1: ["left", "right", "up", "down", "left", "win"],
    2: ["right", "up", "down"],
    3: ["left", "down", "right", "win"]
}

Storing environment variables:





