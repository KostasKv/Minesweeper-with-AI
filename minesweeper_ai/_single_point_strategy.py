''' Scrap code from NoUncessaryGuessSolver. Probably needs some fixing to get it to work. '''

# def singlePointStrategy(self, sample):
#         all_sure_moves_found = set()
#         tiles_and_adjacents_of_interest = []

#         for (sample_y, row) in enumerate(sample[1 : -1]):
#             for (sample_x, tile) in enumerate(row[1 : -1]):
#                 # Skip tiles that can't be used to determine if neighbouring
#                 # tiles are/aren't mines using SPS.
#                 if not tile or not tile.uncovered:
#                     continue

#                 adjacent_tiles = self.getAdjacentTilesInSample((sample_x, sample_y), sample)
#                 self.cheekyHighlight(adjacent_tiles)
#                 sure_moves = self.singlePointStrategyOnTileAndAdjacents(tile, adjacent_tiles)

#                 if sure_moves:
#                     all_sure_moves_found.update(sure_moves)
#                 else:
#                     # Incase SPS needs to be repeated multiple times for this sample.
#                     # Only uncovered inside tiles which haven't had sure moves yet
#                     # could possible have sure moves later (after nearby sure moves are found first)
#                     tiles_and_adjacents_of_interest.append((tile, adjacent_tiles))
#                 self.removeHighlight(adjacent_tiles)

#         moves_found = (len(all_sure_moves_found) > 0)

#         # Sure moves found after an iteration can lead to the discovery of new sure moves in the same sample.
#         # Therefore, SPS should be repeated a bunch until it can no longer find any more sure moves in the sample.
#         while moves_found:
#             moves_found = False

#             for (tile, adjacent_tiles) in tiles_and_adjacents_of_interest:
#                 sure_moves = self.singlePointStrategyOnTileAndAdjacents(tile, adjacent_tiles)

#                 if sure_moves:
#                     moves_found = True
#                     all_sure_moves_found.update(sure_moves)

#                     # Once sure moves are found around a tile, it can't give us any more sure moves.
#                     tiles_and_adjacents_of_interest.remove((tile, adjacent_tiles))

#         return all_sure_moves_found

#     '''
#         Side effect: for every solution it finds, this method marks the tile with
#         that solution. This means the sample can be affected.
#     '''
#     def singlePointStrategyOnTileAndAdjacents(self, tile, adjacent_tiles):
#         sure_moves = set()
#         num_flagged = 0
#         adjacent_covered_tiles = []

#         for adjacent in adjacent_tiles:
#             if adjacent.uncovered:
#                 continue

#             if adjacent.is_flagged:
#                 num_flagged += 1
#             else:
#                 adjacent_covered_tiles.append(adjacent)

#         if adjacent_covered_tiles:
#             self.cheekyHighlight(tile, 4)
#             self.cheekyHighlight(adjacent_covered_tiles, 1)

#             adjacent_mines_not_flagged = tile.num_adjacent_mines - num_flagged

#             if adjacent_mines_not_flagged == 0:
#                 sure_moves = self.formIntoSureMovesAndUpdateTilesWithSolution(adjacent_covered_tiles, is_mine=True)
#             elif adjacent_mines_not_flagged == len(adjacent_covered_tiles):
#                 sure_moves = self.formIntoSureMovesAndUpdateTilesWithSolution(adjacent_covered_tiles, is_mine=True)

#             self.removeHighlight(tile, 4)
#             self.removeHighlight(adjacent_covered_tiles, 1)

#         # # DEBUG
#         # for (x, y, is_mine) in sure_moves_found:
#         #     if is_mine:
#         #         code = 12
#         #     else:
#         #         code = 11

#         #     self.removeHighlight((x, y), code)

#         return sure_moves

#     def formIntoSureMovesAndUpdateTilesWithSolution(self, adjacent_covered_tiles, is_mine=True):
#         sure_moves = set()

#         for tile in adjacent_covered_tiles:
#             move = (tile.x, tile.y, is_mine)
#             sure_moves.add(move)

#             # Mark solution on sample's tile itself
#             if isinstance(tile, SampleOutsideTile):
#                 tile.setIsMine(is_mine)
#             else:
#                 tile.is_flagged = is_mine

#             # DEBUG
#             if is_mine:
#                 code = 12
#             else:
#                 code = 11
#             self.cheekyHighlight(tile, code)

#         return sure_moves

#     @staticmethod
#     def getAdjacentTilesInSample(tile_sample_coords, sample, include_outside=False):
#         max_x = len(sample[0]) - 1
#         max_y = len(sample) - 1

#         (x, y) = tile_sample_coords
#         adjacents = []
#         is_outside = False

#         for i in [-1, 0, 1]:
#             new_x = x + i
            
#             if new_x < 0 or new_x > max_x:
#                 if include_outside:
#                     is_outside = True
#                 else:
#                     continue

#             for j in [-1, 0, 1]:
#                 new_y = y + j

#                 if new_y < 0 or new_y > max_y:
#                     if include_outside:
#                         is_outside = True
#                     else:
#                         continue

#                 # We want adjacent tiles, not the tile itself
#                 if new_x == x and new_y == y:
#                     continue
                
#                 if is_outside:
#                     adjacent = (new_x, new_y)
#                 else:
#                     adjacent = sample[new_y][new_x]

#                 adjacents.append(adjacent)

#         return adjacents

#     def updateSampleWithSureMoves(sample, sure_moves_found):
#         for (x, y, is_mine) in sure_moves_found:
#             sample[y][x].setIsMine(is_mine)

#         return sample