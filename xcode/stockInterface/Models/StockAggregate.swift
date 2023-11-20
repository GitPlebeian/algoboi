//
//  StockAggregate.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import Foundation

struct StockAggregate: Codable {
    
    let symbol: String
    var name: String
    var candles: [Candle] = []
    
    init(symbol: String, candles: [Candle], name: String = "") {
        self.symbol = symbol
        self.candles = candles
        self.name = name
    }
    
    init?(data: Data) {
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                guard let symbol = json["symbol"] as? String else {return nil}
                guard let bars = json["bars"] as? [[String: Any]] else { return nil }
                var candles: [Candle] = []
                for bar in bars {
                    let volume = bar["v"] as! Int64
                    let volumeWeighted = bar["vw"] as! Double
                    let timestamp = bar["t"] as! String
                    let date = DateFormatter.AlpacaDateFormatter.date(from: timestamp)!
                    let transactionCount = bar["n"] as! Int64
                    let open = bar["o"] as! Double
                    let close = bar["c"] as! Double
                    let high = bar["h"] as! Double
                    let low = bar["l"] as! Double
                    let candle = Candle(volume: volume,
                                        volumeWeighted: Float(volumeWeighted),
                                        timestamp: date,
                                        transactionCount: transactionCount, open: Float(open),
                                        close: Float(close),
                                        high: Float(high),
                                        low: Float(low))
                    candles.append(candle)
                }
                self.init(symbol: symbol, candles: candles)
            } else {
                return nil
            }
        } catch let e {
            print("Error: \(e)")
            return nil
        }
    }
}
