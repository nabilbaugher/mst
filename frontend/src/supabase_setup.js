import { uuid } from '@supabase/gotrue-js/dist/module/lib/helpers'
import { createClient } from '@supabase/supabase-js'

const uuidv4 = require("uuid/v4")

const supabaseUrl = process.env.SUPABASE_URL
const supabaseKey = process.env.SUPABASE_KEY
const supabase = createClient(supabaseUrl, supabaseKey)

// Test on supabase

const { error } = await supabase
    .from('keystroke_sequence')
    .insert({ 
        tester_id: uuid.v4(), 
        created_at: Date.now(), 
        keystroke_sequences: {
            1: ["left", "right", "up", "down", "left", "win"],
            2: ["right", "up", "down"],
            3: ["left", "down", "right", "win"]
    }
})