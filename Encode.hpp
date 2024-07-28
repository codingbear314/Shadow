#pragma once

/**
* Shadow C++ / Encode.hpp
*
* Part of the Shadow C++ chess engine project.
*
* 2024, js314
*/

#include <torch/torch.h>
#include <intrin.h>
# include "chess.hpp"

torch::Tensor encode_board(const chess::Board& board)
{
	torch::Tensor board_tensor = torch::zeros({ 8, 8, 6 }, torch::kFloat32)
		
#ifdef SDW_USE_CUDA
		.to(torch::kCUDA);
#else
		.to(torch::kCPU);
#endif

	uint64_t P = board.pieces(chess::PieceType::PAWN, chess::Color::WHITE).getBits();
	uint64_t N = board.pieces(chess::PieceType::KNIGHT, chess::Color::WHITE).getBits();
	uint64_t B = board.pieces(chess::PieceType::BISHOP, chess::Color::WHITE).getBits();
	uint64_t R = board.pieces(chess::PieceType::ROOK, chess::Color::WHITE).getBits();
	uint64_t Q = board.pieces(chess::PieceType::QUEEN, chess::Color::WHITE).getBits();
	uint64_t K = board.pieces(chess::PieceType::KING, chess::Color::WHITE).getBits();
	uint64_t p = board.pieces(chess::PieceType::PAWN, chess::Color::BLACK).getBits();
	uint64_t n = board.pieces(chess::PieceType::KNIGHT, chess::Color::BLACK).getBits();
	uint64_t b = board.pieces(chess::PieceType::BISHOP, chess::Color::BLACK).getBits();
	uint64_t r = board.pieces(chess::PieceType::ROOK, chess::Color::BLACK).getBits();
	uint64_t q = board.pieces(chess::PieceType::QUEEN, chess::Color::BLACK).getBits();
	uint64_t k = board.pieces(chess::PieceType::KING, chess::Color::BLACK).getBits();

	auto process_pieces = [&](uint64_t& pieces, int plane, float value) {
		unsigned long index;
		while (_BitScanReverse64(&index, pieces)) {
			pieces ^= 1ULL << index;
			board_tensor[index / 8][index % 8][plane] = value;
		}
		};

	process_pieces(P, 0, 1.0f);
	process_pieces(p, 0, -1.0f);
	process_pieces(N, 1, 1.0f);
	process_pieces(n, 1, -1.0f);
	process_pieces(B, 2, 1.0f);
	process_pieces(b, 2, -1.0f);
	process_pieces(R, 3, 1.0f);
	process_pieces(r, 3, -1.0f);
	process_pieces(Q, 4, 1.0f);
	process_pieces(q, 4, -1.0f);

	unsigned long indexK, indexk;
	_BitScanReverse64(&indexK, K);
	_BitScanReverse64(&indexk, k);
	board_tensor[indexK / 8][indexK % 8][5] = 1.0f;
	board_tensor[indexk / 8][indexk % 8][5] = -1.0f;
	board_tensor[indexk / 8][indexk % 8][5] = -1.0f;

	return board_tensor;
}

int _SHDW_Sub_del_to_uns(const int x) {
	// convert -7 to 7 to 0 to 13, except 0
	return x + 7 - (x > 0 ? 1 : 0);
}

int _SHDW_Sub_uns_to_del(const int x) {
	// convert 0 to 13 into -7 to 7
	return x - 7 + // 0 to 13 to -7 to 6
		(x >= 7 ? 1 : 0); // 0 to 6 to 0 to 7
}

int EncodeMove(const chess::Move& move)
{
	chess::Square from = move.from();
	chess::Square to = move.to();
	const int from_encoded = (from.file() + from.rank() * 8) * 73; // Max (7 + 7 * 8) * 73 = 4599

	const int FF = from.file();
	const int FR = from.rank();
	const int TF = to.file();
	const int TR = to.rank();

	const int deltaFile = TF - FF;
	const int deltaRank = TR - FR;

	if (move.typeOf() == chess::Move::PROMOTION) {
		if (move.promotionType() == chess::PieceType::KNIGHT) {
			return 64 + from_encoded
				+ (deltaFile + 1 /*0 to 2, depending on capture direction*/);
		}
		if (move.promotionType() == chess::PieceType::BISHOP) {
			return 64 + from_encoded
				+ 3
				+ (deltaFile + 1 /*0 to 2, depending on capture direction*/);
		}
		if (move.promotionType() == chess::PieceType::ROOK) {
			return 64 + from_encoded // 64 for other moves
				+ 6
				+ (deltaFile + 1 /*0 to 2, depending on capture direction*/); // Max value, 64 + 6 + 2= 72
			// Max return = 4599 + 72 = 4671
		}
		// Queen promotion will be considered a queen move, so we don't consider it here.
	}

	// Rook move
	if (deltaFile == 0) {
		// Delta rank can be -7 to 7, except 0
		// (deltaRank + 7 - (deltaRank > 0 ? 1 : 0)) is 0 to 13
		return from_encoded + _SHDW_Sub_del_to_uns(deltaRank);
		// from encoded is 0 to 63 * 73, rest is 0 to 13
	}
	if (deltaRank == 0) {
		// Delta file can be -7 to 7, except 0
		// (deltaFile + 7 - (deltaFile > 0 ? 1 : 0)) is 0 to 13
		return from_encoded + 14 + _SHDW_Sub_del_to_uns(deltaFile);
		// from encoded is 0 to 63 * 73, rest is 14 to 27
	}

	if (deltaFile == deltaRank) {
		// Delta file can be -7 to 7, except 0
		// (deltaFile + 7 - (deltaFile > 0 ? 1 : 0)) is 0 to 13
		return from_encoded + 28 + _SHDW_Sub_del_to_uns(deltaFile);
		// from encoded is 0 to 63 * 73, rest is 28 to 41
	}
	if (deltaFile == -deltaRank) {
		// Delta file can be -7 to 7, except 0
		// (deltaFile + 7 - (deltaFile > 0 ? 1 : 0)) is 0 to 13
		return from_encoded + 42 + _SHDW_Sub_del_to_uns(deltaFile);
		// from encoded is 0 to 63 * 73, rest is 42 to 55
	}

	// Knight move
	// Simply do 8 move check
	if (deltaFile == 2 && deltaRank == 1)
		return from_encoded + 56;
	if (deltaFile == 2 && deltaRank == -1)
		return from_encoded + 57;
	if (deltaFile == -2 && deltaRank == 1)
		return from_encoded + 58;
	if (deltaFile == -2 && deltaRank == -1)
		return from_encoded + 59;
	if (deltaFile == 1 && deltaRank == 2)
		return from_encoded + 60;
	if (deltaFile == 1 && deltaRank == -2)
		return from_encoded + 61;
	if (deltaFile == -1 && deltaRank == 2)
		return from_encoded + 62;
	if (deltaFile == -1 && deltaRank == -2)
		return from_encoded + 63;

	// The code shouldn't reach here.
	std::cout << "Invalid move, " << move << std::endl;
	assert(false);
}

chess::Move DecodeMove(const int move_int, const bool isPawnMove) {
	const int from_square = move_int / 73;
	const auto fromSQ = chess::Square(
		chess::File(from_square % 8),
		chess::Rank(from_square / 8)
	);

	const int moveType = move_int % 73;

	int ToFile = from_square % 8;
	int ToRank = from_square / 8;

	if (moveType < 14) { // Delta File = 0, Rook move
		// (moveTyoe - 7) is -7 to 6, map 0~6 to 1~7
		const int mappedDeltaRank = _SHDW_Sub_uns_to_del(moveType);
		ToRank += mappedDeltaRank;
	}
	else if (moveType < 28) { // Delta Rank = 0, Rook move
		const int mappedDeltaFile = _SHDW_Sub_uns_to_del(moveType - 14);
		ToFile += mappedDeltaFile;
	}
	else if (moveType < 42) { // Delta File = Delta Rank, Bishop move
		const int mappedDelta = _SHDW_Sub_uns_to_del(moveType - 28);
		ToFile += mappedDelta;
		ToRank += mappedDelta;
	}
	else if (moveType < 56) { // Delta File = -Delta Rank, Bishop move
		const int mappedDelta = _SHDW_Sub_uns_to_del(moveType - 42);
		ToFile += mappedDelta;
		ToRank -= mappedDelta;
	}
	else if (moveType < 64) {
		// Knight move
		switch (moveType) {
		case 56: ToFile += 2; ToRank += 1; break;
		case 57: ToFile += 2; ToRank -= 1; break;
		case 58: ToFile -= 2; ToRank += 1; break;
		case 59: ToFile -= 2; ToRank -= 1; break;
		case 60: ToFile += 1; ToRank += 2; break;
		case 61: ToFile += 1; ToRank -= 2; break;
		case 62: ToFile -= 1; ToRank += 2; break;
		case 63: ToFile -= 1; ToRank -= 2; break;
		default: assert(false);
		}
	}
	else { // Underpromotion, 64 ~ 72
		assert(isPawnMove);
		if (ToRank == 1) ToRank = 0;
		if (ToRank == 6) ToRank = 7;
		const int PromotionType = (moveType - 64) / 3;
		// Knight, bishop, rook
		// 0, 1, 2
		const int DeltaFile = (moveType - 64) % 3 - 1;

		switch (PromotionType) {
		case 0:
			return chess::Move::make<chess::Move::PROMOTION>
				(
					fromSQ,
					chess::Square(
						chess::File(ToFile + DeltaFile),
						chess::Rank(ToRank)
					),
					chess::PieceType::KNIGHT
				);
		case 1:
			return chess::Move::make<chess::Move::PROMOTION>
				(
					fromSQ,
					chess::Square(
						chess::File(ToFile + DeltaFile),
						chess::Rank(ToRank)
					),
					chess::PieceType::BISHOP
				);
		case 2:
			return chess::Move::make<chess::Move::PROMOTION>
				(
					fromSQ,
					chess::Square(
						chess::File(ToFile + DeltaFile),
						chess::Rank(ToRank)
					),
					chess::PieceType::ROOK
				);
		}
	}

	// Promotion to queen
	if (isPawnMove && (ToRank == 0 || ToRank == 7)) {
		return chess::Move::make<chess::Move::PROMOTION>
			(
				fromSQ,
				chess::Square(
					chess::File(ToFile),
					chess::Rank(ToRank)
				),
				chess::PieceType::QUEEN
			);
	}

	return chess::Move::make(
		fromSQ,
		chess::Square(
			chess::File(ToFile),
			chess::Rank(ToRank)
		)
	);
}

chess::Move MirrorMove(chess::Move mv) {
	if (mv.typeOf() == chess::Move::PROMOTION) {
		return chess::Move::make<chess::Move::PROMOTION>(
			chess::Square(
				mv.from().file(),
				chess::Rank(
					7 - (int)(mv.from().rank())
				)
			),
			chess::Square(
				mv.to().file(),
				chess::Rank(
					7 - (int)(mv.to().rank())
				)
			),
			mv.promotionType()
		);
	}
	else {
		return chess::Move::make(
			chess::Square(
				mv.from().file(),
				chess::Rank(
					7 - (int)(mv.from().rank())
				)
			),
			chess::Square(
				mv.to().file(),
				chess::Rank(
					7 - (int)(mv.to().rank())
				)
			)
		);
	}
}