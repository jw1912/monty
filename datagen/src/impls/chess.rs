use bulletformat::{BulletFormat, ChessBoard};
use monty::{chess::Chess, GameRep};

use crate::{DatagenSupport, PolicyFormat};

#[derive(Clone, Copy, Default)]
pub struct ChessMoveInfo {
    pub mov: u16,
    pub visits: i16,
}

pub struct ChessPolicyData {
    board: ChessBoard,
    moves: [ChessMoveInfo; 118],
    num: usize,
}

impl PolicyFormat<Chess> for ChessPolicyData {
    const MAX_MOVES: usize = 118;

    fn push(&mut self, mov: <Chess as GameRep>::Move, visits: i16) {
        let from = u16::from(mov.from()) << 10;
        let to = u16::from(mov.to()) << 4;

        self.moves[self.num] = ChessMoveInfo {
            mov: from | to | u16::from(mov.flag()),
            visits,
        };

        self.num += 1;
    }

    fn set_result(&mut self, result: f32) {
        self.board.set_result(result);
    }
}

impl DatagenSupport for Chess {
    type MoveInfo = ChessMoveInfo;
    type ValueData = ChessBoard;
    type PolicyData = ChessPolicyData;

    fn into_policy(pos: &Self, score: f32) -> Self::PolicyData {
        let stm = pos.stm();
        let bbs = pos.bbs();

        let mut score = -(400.0 * (1.0 / score.clamp(0.03, 0.97) - 1.0).ln()) as i16;

        if pos.stm() == 1 {
            score = -score;
        }

        ChessPolicyData {
            board: ChessBoard::from_raw(bbs, stm, score, 0.5).unwrap(),
            moves: [ChessMoveInfo::default(); 118],
            num: 0,
        }
    }

    fn into_value(pos: &Self, score: f32) -> Self::ValueData {
        let stm = pos.stm();
        let bbs = pos.bbs();

        let mut score = -(400.0 * (1.0 / score.clamp(0.03, 0.97) - 1.0).ln()) as i16;

        if pos.stm() == 1 {
            score = -score;
        }

        ChessBoard::from_raw(bbs, stm, score, 0.5).unwrap()
    }
}
