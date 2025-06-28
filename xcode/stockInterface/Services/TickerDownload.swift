//
//  TickerDownload.swift
//  stockInterface
//
//  Created by CHONK on 9/29/23.
//

import CoreData

class TickerDownload {
    
    static let shared = TickerDownload()
    
    var lastDownload: DispatchTime?
    var last12SecondDownload: DispatchTime?
    
    static let finnhubAPIKey: String = "cdgmsniad3i2r375a9p0cdgmsniad3i2r375a9pg"
    
    // MARK: Public
    
    // Get Market Cap
//    https://finnhub.io/api/v1/stock/profile2?symbol=AAPL&token=cdgmsniad3i2r375a9p0cdgmsniad3i2r375a9pg
    func getMarketCap(ticker: String, _ completionHandler: @escaping (((String, String, String, Double)?) -> Void)) {
        
        let urlString = "https://finnhub.io/api/v1/stock/profile2?symbol=\(ticker.uppercased())&token=cdgmsniad3i2r375a9p0cdgmsniad3i2r375a9pg"
        guard let url = URL(string: urlString) else {
            completionHandler(nil)
            return
        }
        let request = URLRequest(url: url)
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completionHandler(nil)
                print(error)
                return
            }
            
            guard let data = data else {
                completionHandler(nil)
                return
            }
            do {
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {completionHandler(nil); return}
                guard let symbol = json["ticker"] as? String else {
                    completionHandler(nil)
                    return
                }
                guard let companyName = json["name"] as? String else {
                    completionHandler(nil)
                    return
                }
                guard let industry = json["finnhubIndustry"] as? String else {
                    completionHandler(nil)
                    return
                }
                guard let marketCap = json["marketCapitalization"] as? Double else {
                    completionHandler(nil)
                    return
                }
                completionHandler((symbol, companyName, industry, marketCap))
            } catch let e {
                print("Error: \(e)")
                completionHandler(nil)
            }
        }.resume()
    }
    
    // MARK: Get Minute Interval
    
    func getAlpacaStockMinuteInterval(runInBulk: Bool = false,
                                      ticker: String,
                                      date: (Int, Int, Int),
                                      _ completionHandler: @escaping ((String, StockAggregate?) -> Void)) {
        
//        let dateFormatter = DateFormatter()
//        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss-05:00"
//        dateFormatter.timeZone = TimeZone(identifier: "GMT")
        
        guard let startDate = Calendar.current.date(from: DateComponents(year: date.2, month: date.0, day: date.1, hour: 8, minute: 30)) else {
            completionHandler("Unable to configure date from components", nil)
            return
        }
        let endDate = Calendar.current.date(byAdding: DateComponents(hour: 6, minute: 29), to: startDate)!
//        print(startDate, endDate)
//        print(dateFormatter.string(from: startDate), dateFormatter.string(from: endDate))
//        let urlString = "https://data.alpaca.markets/v2/stocks/\(ticker.uppercased())/bars?timeframe=1Min&adjustment=all&start=\(dateFormatter.string(from: startDate))&end=\(dateFormatter.string(from: endDate))&limit=5000"
        let urlString = "https://data.alpaca.markets/v2/stocks/\(ticker.uppercased())/bars?timeframe=1Min&adjustment=all&start=\(dateFormatter.string(from: startDate))&end=\(dateFormatter.string(from: endDate))&limit=5000"
//        let urlString = "https://data.alpaca.markets/v2/stocks/\(ticker.uppercased())/bars?timeframe=1Min&adjustment=all&limit=5000"
//        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        guard let url = URL(string: urlString) else {
            completionHandler("Unable to create url", nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("6BhKowZO30rHQ3C7M6w00C5pjGlT5PWl70gKsFag", forHTTPHeaderField: "APCA-API-SECRET-KEY")
        request.addValue("AKW75YXKPL201JU3WWFK", forHTTPHeaderField: "APCA-API-KEY-ID")
        
        if runInBulk == false {
            if let lastDownload = lastDownload {
                let now = DispatchTime.now()
                let difference = (now.uptimeNanoseconds - lastDownload.uptimeNanoseconds) / 1000000
                if difference < 350 {
                    Thread.sleep(forTimeInterval: TimeInterval(350 - difference) / 1000)
                }
                self.lastDownload = DispatchTime.now()
            } else {
                lastDownload = DispatchTime.now()
            }
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error getting data: \(error)")
                completionHandler("Error during request", nil)
                return
            }
            
            guard let data = data else {
                completionHandler("Data returned == nil", nil)
                return
            }
            guard let response = response else {
                completionHandler("No Response == nil", nil)
                return
            }
            guard let httpResponse = response as? HTTPURLResponse else {
                completionHandler("Cannot convert reponse to HTTP response", nil)
                return
            }
            if httpResponse.statusCode == 429 {
                print("💩💩💩 Too Many Requests")
            }
            if httpResponse.statusCode == 422 {
                print(httpResponse.statusCode)
            }
             guard let stockAggregate = StockAggregate(data: data) else {
                completionHandler("Could not decode data during StockAggregate Initialization", nil)
                return
            }
            completionHandler("Success", stockAggregate)
        }.resume()
    }
    
    // Get Alpaca Data
    /// dateComponents = month,day,year
    func getAlpacaStock(runInBulk: Bool = false, dateComponents: (Int, Int, Int)? = nil, ticker: String, year: Int? = nil, _ completionHandler: @escaping ((String, StockAggregate?) -> Void)) {
        
//        let b = StockAggre
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss-05:00"
        
        let date = Date()
        
        var startDate: Date?
        
        if let dateComponents = dateComponents {
            guard let d = Calendar.current.date(from: DateComponents(year: dateComponents.2, month: dateComponents.0, day: dateComponents.1)) else {
                completionHandler("Unable to configure date from components", nil)
                return
            }
            let dayComp = DateComponents(day: -1)
            let dd = Calendar.current.date(byAdding: dayComp, to: d)
            startDate = dd
        }
        
        if let year = year {
            let yearComp = DateComponents(year: -year)
            guard let d = Calendar.current.date(byAdding: yearComp, to: date) else {
                completionHandler("Unable to configure date", nil)
                return
            }
            startDate = d
        }
        
        guard let startDate = startDate else {
            completionHandler("No Start Date", nil)
            return
        }
        
        
        let urlString = "https://data.alpaca.markets/v2/stocks/\(ticker.uppercased())/bars?timeframe=1Day&adjustment=all&start=\(dateFormatter.string(from: startDate))&limit=10000"
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
        guard let url = URL(string: urlString) else {
            completionHandler("Unable to create url", nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("6BhKowZO30rHQ3C7M6w00C5pjGlT5PWl70gKsFag", forHTTPHeaderField: "APCA-API-SECRET-KEY")
        request.addValue("AKW75YXKPL201JU3WWFK", forHTTPHeaderField: "APCA-API-KEY-ID")
        
        if runInBulk == false {
            if let lastDownload = lastDownload {
                let now = DispatchTime.now()
                let difference = (now.uptimeNanoseconds - lastDownload.uptimeNanoseconds) / 1000000
                if difference < 350 {
                    Thread.sleep(forTimeInterval: TimeInterval(350 - difference) / 1000)
                }
                self.lastDownload = DispatchTime.now()
            } else {
                lastDownload = DispatchTime.now()
            }
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Error getting data: \(error)")
                completionHandler("Error during request", nil)
                return
            }
            guard let data = data else {
                completionHandler("Data returned == nil", nil)
                return
            }
            guard let response = response else {
                completionHandler("No Response == nil", nil)
                return
            }
            guard let httpResponse = response as? HTTPURLResponse else {
                completionHandler("Cannot convert reponse to HTTP response", nil)
                return
            }
            if httpResponse.statusCode == 429 {
                print("💩💩💩 Too Many Requests")
            }
            guard let stockAggregate = StockAggregate(data: data) else {
                completionHandler("Could not decode data during StockAggregate Initialization", nil)
                guard let dataString = String(data: data, encoding: .utf8) else {
                    print("Could not get string form data value")
                    return
                }
                print("Could not decode data during StockAggregate Initialization: \(dataString)")
                return
            }
            completionHandler("Success", stockAggregate)
        }.resume()
    }
    
    func getEveryTicker(nextURLString: String? = nil, handler: @escaping ([TickerNameModel]?, Bool) -> Bool) {
        
        var urlString: String
        
        if let nextURLString = nextURLString {
            urlString = nextURLString + "&apiKey=Mnd1Ub6_2gZs2_2SLdQsdGVj_ZjribwsSbjFA4"
        } else {
            urlString = "https://api.polygon.io/v3/reference/tickers?type=CS&market=stocks&active=true&&limit=1000&apiKey=Mnd1Ub6_2gZs2_2SLdQsdGVj_ZjribwsSbjFA4"
        }
        
        let request = URLRequest(url: URL(string: urlString)!)
        self.last12SecondDownload = DispatchTime.now()
        URLSession.shared.dataTask(with: request) {data, response, error in
            if let error = error {
                let _ = handler(nil, false)
                TerminalManager.shared.addText("Error: \(error)", type: .error)
                return
            }
            
            guard let data = data else {
                let _ = handler(nil, false)
                TerminalManager.shared.addText("No data provided", type: .error)
                return
            }
            
            guard let response = response as? HTTPURLResponse else {
                let _ = handler(nil, false)
                TerminalManager.shared.addText("No response provided", type: .error)
                return
            }
            if response.statusCode == 429 {
                let _ = handler(nil, false)
                TerminalManager.shared.addText("Too Many Requests", type: .error)
                return
            }
            do {
                guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    let _ = handler(nil, false)
                    TerminalManager.shared.addText("Error getting Json From Data", type: .error)
                    return
                }
                guard let results = json["results"] as? [[String: Any]] else {
                    let _ = handler(nil, false)
                    TerminalManager.shared.addText("Error getting Json From Data", type: .error)
                    return
                }
                var names: [TickerNameModel] = []
                for e in results {
                    guard let model = TickerNameModel(dict: e) else {
                        continue
                    }
                    names.append(model)
                }
                let shouldContinue = handler(names, false)
                if shouldContinue == false {return}
                guard let nextURL = json["next_url"] as? String else {
                    let _ = handler([], true)
                    return
                }
                let now = DispatchTime.now()
                let difference = (now.uptimeNanoseconds - self.last12SecondDownload!.uptimeNanoseconds) / 1000000
                if difference < 12100 {
                    let sleepAmount = TimeInterval(12100 - difference) / 1000
                    print("Sleeping for \(sleepAmount)")
                    Thread.sleep(forTimeInterval: sleepAmount)
                }
                self.getEveryTicker(nextURLString: nextURL, handler: handler)
            } catch let e {
                let _ = handler(nil, false)
                TerminalManager.shared.addText("Error: \(e)", type: .error)
                return
            }
        }.resume()
    }
    
    // Get Good Tickers From Server
//    func getGoodTickersFromServer(_ completionHandler: @escaping (String?, Int) -> Void) {
//
////        let urlString = "http://192.168.1.117/getGoodStocksList"
//        let urlString = "http://143.198.226.109/getGoodStocksList"
//        guard let url = URL(string: urlString) else {
//            completionHandler("Cannot create url out of string", 0)
//            return
//        }
//
//        let request = URLRequest(url: url)
//        URLSession.shared.dataTask(with: request) { data, response, error in
//            if let error = error {
//                print("Error getting data: \(error)")
//                completionHandler("Error during request", 0)
//                return
//            }
//
//            guard let data = data else {
//                completionHandler("Data returned == nil", 0)
//                return
//            }
//            do {
//                if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
//                    guard let lastScreenTime = json["timeLastScanned"] as? Int else {
//                        completionHandler("Can't get time last scanned", 0)
//                        return
//                    }
//                    guard let isScanning = json["isScanning"] as? Bool else {
//                        completionHandler("Coun't find is scanning", 0)
//                        return
//                    }
//                    var goodStocksData: [StockDataRaw] = []
//                    guard let stockDataStringArray = json["data"] as? String else {
//                        completionHandler("Can't create data string", 0)
//                        return
//                    }
//                    var stockStringArray = stockDataStringArray.components(separatedBy: "\r\n")
//                    stockStringArray.remove(at: stockStringArray.count - 1)
//                    for (index, stockString) in stockStringArray.enumerated() {
//                        guard let data = stockString.data(using: .utf8) else {
//                            completionHandler("Can't create Data", 0)
//                            return
//                        }
//                        guard let stockDataJson = try JSONSerialization.jsonObject(with: data) as? [String: Any]  else {
//                            completionHandler("Can't Create Stock Data Json", 0)
//                            return
//                        }
//
//                        guard let ticker = stockDataJson["ticker"] as? String else {
//                            print("Can't create ticker")
//                            completionHandler("Get get ticker from stock. Index: \(index)", 0)
//                            return
//                        }
//                        guard let barsJson = stockDataJson["bars"] as? [[String: Any]] else  {
//                            print("Create bars array")
//                            completionHandler("Can't create bars array. Index: \(index)", 0)
//                            return
//                        }
//                        var bars: [StockDataPointRaw] = []
//                        for bar in barsJson {
//                            guard let close = bar["close"] as? Double else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            guard let open = bar["open"] as? Double else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            guard let high = bar["high"] as? Double else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            guard let low = bar["low"] as? Double else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            guard let timestamp = bar["timestamp"] as? Int else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            let timestampDate = Date(timeIntervalSince1970: TimeInterval(timestamp / 1000))
//                            guard let volume = bar["volume"] as? Int64 else {
//                                print("Can't create close")
//                                completionHandler("Bad extrapolation", 0)
//                                return
//                            }
//                            bars.append(StockDataPointRaw(volume: volume, volumeWeighted: 0, timestamp: timestampDate, open: Float(open), close: Float(close), high: Float(high), low: Float(low), transactionCount: 0))
//                        }
//                        goodStocksData.append(StockDataRaw(ticker: ticker, dataPoints: bars))
//                    }
//                    StockDataController.shared.isScanning = isScanning
//                    StockDataController.shared.lastScanDate = Date(timeIntervalSince1970: TimeInterval(lastScreenTime / 1000))
//                    StockDataController.shared.setAllGoodStocks(goodStocksData)
//                    completionHandler(nil, goodStocksData.count)
//                }
//            } catch let e {
//                print("error: \(e)")
//                completionHandler("Error during decoding", 0)
//            }
//
//        }.resume()
//    }
    
    // Get Ticker Data
//    func getTickerData(ticker: String, _ completionHandler: @escaping ((Bool, String) -> Void)) {
        
//        if StockDataController.shared.isStockAlreadySaved(ticker: ticker) {
//            completionHandler(false, "Stock already saved")
//            return
//        }
//
//        let date = Date()
//        let yearComp = DateComponents(year: -4)
//        guard let previousDate = Calendar.current.date(byAdding: yearComp, to: date) else {
//            completionHandler(false, "Unable to configure dates")
//            return
//        }
//
//        let formatter = DateFormatter()
//        formatter.dateFormat = "yyyy-MM-dd"
//
//
//        let urlString = "https://api.polygon.io/v2/aggs/ticker/" + ticker.uppercased() + "/range/1/day/" + formatter.string(from: previousDate) + "/" + formatter.string(from: date) + "?adjusted=true&sort=asc&limit=2000&apiKey=Mnd1Ub6_2gZs2_2SLdQsdGVj_ZjribwsSbjFA4"
//
//        guard let url = URL(string: urlString) else {
//            completionHandler(false, "Unable to create URL String")
//            return
//        }
//        print("Sending request to server")
//        URLSession.shared.dataTask(with: url) { data, response, error in
//            if let error = error {
//                completionHandler(false, "Error Fetching Data: \(error.localizedDescription)")
//                return
//            }
//            guard let data = data else {
//                completionHandler(false, "Didn't return any data from server")
//                return
//            }
//            do {
//
//                let ticker = try JSONDecoder().decode(StockData.self, from: data)
//                print("Save ticker with \(ticker.dataPoints.count) datapoints")
//                CoreDataStack.saveToPersistentStore()
//                StockDataController.shared.setStockData()
//                completionHandler(true, "Data Saved Successfully")
//            } catch let e {
//                print(e)
//                completionHandler(false, "Unable to decode data")
//            }
//        }.resume()
//    }
    
//    // Get CSV Data
//    func getCSVData(_ data: Data) {
//        guard let stringData = String(data: data, encoding: .utf8) else {return}
//
//        let lines = stringData.split(separator: "\n")
//
//        var tickers: [Ticker] = []
//
//        for line in lines[1...] {
//            let columns = line.split(separator: ",")
//            if columns.count < 6 {
//                continue
//            }
//            if (columns[3] != "NYSE" && columns[3] != "NASDAQ") || columns[4] != "USD" || columns[5] != "\"Common Stock\"" {
//                continue
//            }
//
//            let symbol =   String(columns[0])
//            let name   =   String(columns[1])
//            let exchange = String(columns[3])
//            tickers.append(Ticker(ticker: symbol,
//                                  name: name,
//                                  exchange: exchange))
//        }
//
//        AllStocksListController.shared.updateTickers(tickers)
//
//    }
}
