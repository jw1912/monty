mod attacks;
mod board;
mod consts;
mod moves;

pub use board::Board;
pub use moves::Move;

use crate::{
    comm::UciLike,
    game::{GameRep, GameState},
    value::{ValueNetwork, ValueFeatureMap},
    MctsParams,
};

const STARTPOS: &str = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1";

static NET: ValueNetwork<768, 8> =
    unsafe { std::mem::transmute(*include_bytes!("../resources/shatranj-value002.bin")) };

impl ValueFeatureMap for Board {
    fn value_feature_map<F: FnMut(usize)>(&self, f: F) {
        self.features_map(f);
    }
}

pub struct Uci;
impl UciLike for Uci {
    const NAME: &'static str = "uci";
    const NEWGAME: &'static str = "ucinewgame";
    const OK: &'static str = "uciok";
    const FEN_STRING: &'static str = include_str!("../resources/chess-fens.txt");

    type Game = Shatranj;

    fn options() {
        println!("option name UCI_Variant type combo default shatranj var shatranj");
    }
}

#[derive(Clone)]
pub struct Shatranj {
    board: Board,
    stack: Vec<u64>,
}

impl Default for Shatranj {
    fn default() -> Self {
        let board = Board::parse_fen(STARTPOS);

        Self {
            board,
            stack: Vec::new(),
        }
    }
}

impl Shatranj {
    pub fn bbs(&self) -> [u64; 8] {
        self.board.bbs()
    }

    pub fn board(&self) -> Board {
        self.board
    }
}

impl GameRep for Shatranj {
    type Move = Move;

    const STARTPOS: &'static str = STARTPOS;

    const MAX_MOVES: usize = 512;

    fn default_mcts_params() -> MctsParams {
        MctsParams::default()
    }

    fn is_same(&self, other: &Self) -> bool {
        self.board == other.board
    }

    fn conv_mov_to_str(&self, mov: Self::Move) -> String {
        mov.to_uci()
    }

    fn from_fen(fen: &str) -> Self {
        Self {
            board: Board::parse_fen(fen),
            stack: Vec::new(),
        }
    }

    fn map_legal_moves<F: FnMut(Self::Move)>(&self, mut f: F) {
        self.board.map_legal_moves(&mut f);
    }

    fn game_state(&self) -> GameState {
        self.board.game_state(&self.stack)
    }

    fn make_move(&mut self, mov: Self::Move) {
        self.stack.push(self.board.hash());
        self.board.make(mov);

        if self.board.halfm() == 0 {
            self.stack.clear();
        }
    }

    fn stm(&self) -> usize {
        self.board.stm()
    }

    fn tm_stm(&self) -> usize {
        self.stm()
    }

    fn get_policy_feats(&self) -> goober::SparseVector {
        goober::SparseVector::with_capacity(0)
    }

    fn get_policy(&self, _: Self::Move, _: &goober::SparseVector) -> f32 {
        0.0
    }

    fn get_value(&self) -> i32 {
        NET.eval(&self.board)
    }

    fn perft(&self, depth: usize) -> u64 {
        perft::<true, true>(&self.board, depth as u8)
    }
}

impl std::fmt::Display for Shatranj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let feats = self.get_policy_feats();
        let mut moves = Vec::new();
        let mut max = f32::NEG_INFINITY;
        self.map_legal_moves(|mov| {
            let policy = self.get_policy(mov, &feats);
            moves.push((mov, policy));

            if policy > max {
                max = policy;
            }
        });

        let mut total = 0.0;

        for (_, policy) in moves.iter_mut() {
            *policy = (*policy - max).exp();
            total += *policy;
        }

        for (_, policy) in moves.iter_mut() {
            *policy /= total;
        }

        let mut w = [0f32; 64];
        let mut count = [0; 64];

        for &(mov, policy) in moves.iter() {
            let fr = usize::from(mov.from());
            let to = usize::from(mov.to());

            w[fr] = w[fr].max(policy);
            w[to] = w[to].max(policy);

            count[fr] += 1;
            count[to] += 1;
        }

        let pcs = [
            ['p', 'n', 'b', 'r', 'q', 'k'],
            ['P', 'N', 'B', 'R', 'Q', 'K'],
        ];

        writeln!(f, "+-----------------+")?;

        for i in (0..8).rev() {
            write!(f, "|")?;

            for j in 0..8 {
                let sq = 8 * i + j;
                let pc = self.board.get_pc(1 << sq);
                let ch = if pc != 0 {
                    let is_white = self.board.piece(0) & (1 << sq) > 0;
                    pcs[usize::from(is_white)][pc - 2]
                } else {
                    '.'
                };

                if count[sq] > 0 {
                    let g = (255.0 * (2.0 * w[sq]).min(1.0)) as u8;
                    let r = 255 - g;
                    write!(f, " \x1b[38;2;{r};{g};0m{ch}\x1b[0m")?;
                } else {
                    write!(f, " \x1b[34m{ch}\x1b[0m")?;
                }
            }

            writeln!(f, " |")?;
        }

        writeln!(f, "+-----------------+")
    }
}

fn perft<const ROOT: bool, const BULK: bool>(pos: &Board, depth: u8) -> u64 {
    let mut count = 0;
    let leaf = depth == 1;

    if BULK && !ROOT && leaf {
        pos.map_legal_moves(&mut |_| count += 1);
    } else {
        pos.map_legal_moves(&mut |mov| {
            let mut tmp = *pos;
            tmp.make(mov);

            let num = if leaf {
                1
            } else {
                perft::<false, BULK>(&tmp, depth - 1)
            };

            count += num;

            if ROOT {
                println!("{}: {num}", mov.to_uci());
            }
        });
    }

    count
}
